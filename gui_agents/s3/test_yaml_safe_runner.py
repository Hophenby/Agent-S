from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
S3_ROOT = REPO_ROOT / "gui_agents" / "s3"
if str(S3_ROOT) not in sys.path:
    sys.path.insert(0, str(S3_ROOT))

from core.observation import Observation
from instruction.yaml.langgraph_instruction_runner import run_safe_instruction_langgraph
from instruction.yaml.yaml_instruction import (
    Actions,
    Images,
    Job,
    Metadata,
    Postcondition,
    Software,
    Step,
    TemplateMatching,
    Transition,
    TransitionCondition,
    YamlInstruction,
)
from instruction.yaml.yaml_instruction_auto_executor import (
    run_safe_instruction,
    verify_step_result,
)
from instruction.yaml.yaml_instruction_parser import load_instruction


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _observation_from_image(image: Image.Image) -> Observation:
    image_bytes = _png_bytes(image)
    return Observation(screenshot=image_bytes, original_screenshot=image_bytes)


def _provider_from_images(images: list[Image.Image]):
    observations = [_observation_from_image(image) for image in images]
    index = {"value": 0}

    def provider():
        position = min(index["value"], len(observations) - 1)
        index["value"] += 1
        return observations[position]

    return provider


def _make_pattern_image() -> Image.Image:
    image = Image.new("RGB", (40, 40), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 31, 31), fill="black")
    draw.rectangle((15, 15, 24, 24), fill="white")
    return image


def _make_canvas_with_pattern(pattern: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (120, 120), "white")
    canvas.paste(pattern, (30, 30))
    return canvas


def _write_image(path: Path, image: Image.Image) -> str:
    image.save(path)
    return str(path)


def _make_instruction(*steps: Step) -> YamlInstruction:
    return YamlInstruction(
        name="test",
        metadata=Metadata(title="test", software=Software(name="app", version="1")),
        on={"workflow_dispatch": None},
        jobs={
            "instruction_flow": Job(
                name="instruction_flow",
                runs_on="human-desktop",
                steps=list(steps),
            )
        },
    )


def test_load_instruction_parses_postconditions_and_resolves_paths():
    instruction = load_instruction(
        REPO_ROOT / "gui_agents" / "s3" / "instruction" / "yaml" / "example_instruction.yaml"
    )

    step = next(item for item in instruction.steps if item.id == "step_3_find_full_text")

    assert instruction.metadata.source_markdown is not None
    assert Path(instruction.metadata.source_markdown).is_absolute()
    assert step.postconditions
    assert step.postconditions[0].id == "submenu_visible"
    assert step.transitions[0].when.matched_postcondition == "submenu_visible"


def test_verify_step_result_matches_named_postcondition(tmp_path: Path):
    pattern = _make_pattern_image()
    screenshot = _make_canvas_with_pattern(pattern)
    template_path = _write_image(tmp_path / "match.png", pattern)

    step = Step(
        id="step_1",
        name="step_1",
        actions=Actions(),
        images=Images(),
        postconditions=[Postcondition(id="matched", result_image=template_path)],
        template_matching=TemplateMatching(threshold=0.8),
    )

    verification = verify_step_result(step, _observation_from_image(screenshot))

    assert verification.passed is True
    assert verification.status == "success"
    assert verification.matched_postcondition == "matched"
    assert verification.matched_postconditions == ["matched"]


def test_verify_step_result_reports_no_postcondition_matched(tmp_path: Path):
    pattern = _make_pattern_image()
    blank = Image.new("RGB", (120, 120), "white")
    template_path = _write_image(tmp_path / "match.png", pattern)

    step = Step(
        id="step_1",
        name="step_1",
        actions=Actions(),
        images=Images(),
        postconditions=[Postcondition(id="matched", result_image=template_path)],
    )

    verification = verify_step_result(step, _observation_from_image(blank))

    assert verification.passed is False
    assert verification.status == "no_postcondition_matched"
    assert verification.matched_postcondition is None
    assert verification.matched_postconditions == []


def test_verify_step_result_reports_ambiguous_postcondition(tmp_path: Path):
    pattern = _make_pattern_image()
    screenshot = _make_canvas_with_pattern(pattern)
    template_path = _write_image(tmp_path / "match.png", pattern)

    step = Step(
        id="step_1",
        name="step_1",
        actions=Actions(),
        images=Images(),
        postconditions=[
            Postcondition(id="a", result_image=template_path),
            Postcondition(id="b", result_image=template_path),
        ],
    )

    verification = verify_step_result(step, _observation_from_image(screenshot))

    assert verification.passed is False
    assert verification.status == "ambiguous_postcondition"
    assert verification.matched_postcondition is None
    assert verification.matched_postconditions == ["a", "b"]


def test_native_runner_branches_on_no_postcondition_matched(tmp_path: Path):
    pattern = _make_pattern_image()
    blank = Image.new("RGB", (120, 120), "white")
    template_path = _write_image(tmp_path / "match.png", pattern)

    step_1 = Step(
        id="step_1",
        name="step_1",
        actions=Actions(),
        images=Images(),
        postconditions=[Postcondition(id="matched", result_image=template_path)],
        transitions=[
            Transition(
                goto="step_2",
                when=TransitionCondition(status="no_postcondition_matched"),
            )
        ],
    )
    step_2 = Step(id="step_2", name="step_2", actions=Actions(), images=Images())
    provider = _provider_from_images([blank, blank, blank, blank])

    results = run_safe_instruction(
        _make_instruction(step_1, step_2),
        obs=_observation_from_image(blank),
        observation_provider=provider,
    )

    assert [result.step_id for result in results] == ["step_2"]


def test_native_runner_uses_on_failure_for_new_failure_statuses(tmp_path: Path):
    pattern = _make_pattern_image()
    blank = Image.new("RGB", (120, 120), "white")
    template_path = _write_image(tmp_path / "match.png", pattern)

    step_1 = Step(
        id="step_1",
        name="step_1",
        actions=Actions(),
        images=Images(),
        postconditions=[Postcondition(id="matched", result_image=template_path)],
        on_failure="step_2",
    )
    step_2 = Step(id="step_2", name="step_2", actions=Actions(), images=Images())
    provider = _provider_from_images([blank, blank, blank, blank])

    results = run_safe_instruction(
        _make_instruction(step_1, step_2),
        obs=_observation_from_image(blank),
        observation_provider=provider,
    )

    assert [result.step_id for result in results] == ["step_2"]


def test_langgraph_runner_branches_on_ambiguous_postcondition(tmp_path: Path):
    pattern = _make_pattern_image()
    screenshot = _make_canvas_with_pattern(pattern)
    template_path = _write_image(tmp_path / "match.png", pattern)

    step_1 = Step(
        id="step_1",
        name="step_1",
        actions=Actions(),
        images=Images(),
        postconditions=[
            Postcondition(id="a", result_image=template_path),
            Postcondition(id="b", result_image=template_path),
        ],
        transitions=[
            Transition(
                goto="step_2",
                when=TransitionCondition(status="ambiguous_postcondition"),
            )
        ],
    )
    step_2 = Step(id="step_2", name="step_2", actions=Actions(), images=Images())
    provider = _provider_from_images([screenshot, screenshot, screenshot, screenshot])

    result = run_safe_instruction_langgraph(
        _make_instruction(step_1, step_2),
        obs=_observation_from_image(screenshot),
        observation_provider=provider,
    )

    assert result.final_state.get("last_status") == "completed"
    assert result.history == ["verification_failed:step_1", "success:step_2"]
    assert [item.step_id for item in result.results] == ["step_2"]
