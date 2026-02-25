from dataclasses import dataclass

@dataclass
class Observation:
    """
    Represents the observation of the current state of the GUI, including the screenshot and any relevant metadata.
    """

    screenshot:bytes = None  # Compressed Raw bytes of the screenshot image with annotations (e.g., detected elements highlighted)
    original_screenshot:bytes = None  # Uncompressed Raw bytes of the screenshot image

    _others: dict = {}  # Placeholder for any additional metadata or information related to the observation

    _screenshot_b64: str = None  # Base64-encoded string of the screenshot for easy transmission or embedding

    def __setitem__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            self._others[key] = value

    def __getitem__(self, key):
        if hasattr(self, key):
            return super().__getattribute__(key)
        else:
            return self._others.get(key, None)
        
    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            _others = super().__getattribute__("_others")
            if name in _others:
                return _others[name]
            raise AttributeError(f"'Observation' object has no attribute '{name}'")
        
    @property
    def screenshot_width(self):
        return _bytes_to_pil_image(self.screenshot).width
    
    @property
    def screenshot_height(self):
        return _bytes_to_pil_image(self.screenshot).height
    
    @property
    def original_screenshot_width(self):
        return _bytes_to_pil_image(self.original_screenshot).width
    
    @property
    def original_screenshot_height(self):
        return _bytes_to_pil_image(self.original_screenshot).height
    
def _bytes_to_pil_image(image_bytes: bytes):
    from PIL import Image
    from io import BytesIO
    return Image.open(BytesIO(image_bytes))