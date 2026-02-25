"""
Tiny helper to locate a cropped patch inside a full screenshot.

Example
-------
from gui_agents.s3.utils.patch_locator import locate_patch, draw_match_box

result = locate_patch("/path/full.png", "/path/crop.png")
if result:
    print(result)
    preview = draw_match_box("/path/full.png", result)
    preview.save("/path/preview.png")

Notes
-----
- Images should be at the same scale; this simple matcher does not search over scales.
- OpenCV is used when available; the NumPy fallback is O(n*m) and may need a larger `step` on big screenshots.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

try:  # Optional acceleration if OpenCV is available in the environment
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore


ImageLike = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class MatchResult:
    x: int
    y: int
    width: int
    height: int
    score: float
    method: str

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


def _to_gray_array(img: ImageLike) -> np.ndarray:
    """Convert path/PIL/np array to grayscale float array in range [0, 255]."""
    if isinstance(img, (str, Path)):
        pil_img = Image.open(img).convert("L")
        return np.asarray(pil_img, dtype=np.float32)

    if isinstance(img, Image.Image):
        pil_img = img.convert("L")
        return np.asarray(pil_img, dtype=np.float32)

    if isinstance(img, np.ndarray):
        arr = img.astype(np.float32)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            # convert RGB/RGBA to gray using luma weights
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            return 0.299 * r + 0.587 * g + 0.114 * b
    raise TypeError(f"Unsupported image type: {type(img)}")


def _locate_with_cv(full_gray: np.ndarray, patch_gray: np.ndarray) -> Optional[MatchResult]:
    if cv2 is None:
        return None
    res = cv2.matchTemplate(full_gray.astype(np.uint8), patch_gray.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return MatchResult(x=int(max_loc[0]), y=int(max_loc[1]), width=patch_gray.shape[1], height=patch_gray.shape[0], score=float(max_val), method="cv2.TM_CCOEFF_NORMED")


def _locate_with_ncc(full_gray: np.ndarray, patch_gray: np.ndarray, step: int = 1) -> MatchResult:
    ph, pw = patch_gray.shape
    fh, fw = full_gray.shape
    patch_mean = patch_gray.mean()
    patch_std = patch_gray.std() + 1e-6
    patch_norm = (patch_gray - patch_mean) / patch_std

    best_score = float("-inf")
    best_xy = (0, 0)
    for y in range(0, fh - ph + 1, step):
        for x in range(0, fw - pw + 1, step):
            window = full_gray[y : y + ph, x : x + pw]
            w_std = window.std() + 1e-6
            score = float(np.mean((window - window.mean()) * patch_norm) / w_std)
            if score > best_score:
                best_score = score
                best_xy = (x, y)

    return MatchResult(x=best_xy[0], y=best_xy[1], width=pw, height=ph, score=best_score, method="normalized_cross_correlation")


def locate_patch(
    full_image: ImageLike,
    patch_image: ImageLike,
    *,
    prefer_cv2: bool = True,
    step: int = 1,
    score_threshold: float = 0.0,
) -> Optional[MatchResult]:
    """
    Find where a cropped patch best matches inside a full image.

    Args:
        full_image: Path, PIL Image, or ndarray for the full screenshot.
        patch_image: Path, PIL Image, or ndarray for the cropped region.
        prefer_cv2: Use OpenCV matchTemplate if available; otherwise fall back to a NumPy implementation.
        step: Pixel stride for the NumPy fallback (larger values trade accuracy for speed).
        score_threshold: Minimum score required to accept a match. 0 means accept best match regardless.

    Returns:
        MatchResult or None when images are incompatible or the score is below threshold.
    """

    full_gray = _to_gray_array(full_image)
    patch_gray = _to_gray_array(patch_image)

    fh, fw = full_gray.shape
    ph, pw = patch_gray.shape
    if ph > fh or pw > fw:
        return None

    result: Optional[MatchResult] = None
    if prefer_cv2:
        result = _locate_with_cv(full_gray, patch_gray)

    if result is None:
        result = _locate_with_ncc(full_gray, patch_gray, step=step)

    if result.score < score_threshold:
        return None
    return result


def draw_match_box(full_image: ImageLike, match: MatchResult, color: Tuple[int, int, int] = (0, 200, 0), width: int = 3) -> Image.Image:
    """Overlay the located box on top of the original image for quick inspection."""

    img = full_image if isinstance(full_image, Image.Image) else Image.open(full_image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(match.bbox, outline=color, width=width)
    return img
