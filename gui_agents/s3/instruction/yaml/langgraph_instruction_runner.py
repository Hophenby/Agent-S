from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from typing_extensions import TypedDict

from core.observation import Observation
from instruction.yaml.yaml_instruction import Step, YamlInstruction
from instruction.yaml.yaml_instruction_auto_executor import (
    SafeWorkflowError,
    StepExecutionResult,
    StepVerificationError,
    capture_desktop_observation,
    execute_step,
    resolve_next_step_id,
)

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "LangGraph support requires the 'langgraph' package. "
        "Install it with `pip install langgraph`."
    ) from exc


class LangGraphInstructionState(TypedDict, total=False):
    current_step_id: Optional[str]
    last_status: str
    last_error: Optional[str]
    step_counter: int
    max_steps: int


@dataclass
class LangGraphRunResult:
    final_state: Dict[str, Any]
    results: List[StepExecutionResult] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    final_observation: Optional[Observation] = None


class LangGraphInstructionRunner:
    def __init__(
        self,
        instruction: YamlInstruction,
        *,
        template_threshold: float = 0.8,
        observation_provider: Optional[Callable[[], Observation]] = None,
        debug: bool = False,
    ) -> None:
        self.instruction = instruction
        self.steps = instruction.steps
        self.template_threshold = template_threshold
        self.observation_provider = observation_provider
        self.debug = debug

        self.step_lookup: Dict[str, int] = {}
        for index, step in enumerate(self.steps):
            if step.id in self.step_lookup:
                raise SafeWorkflowError(f"Duplicate step id '{step.id}' in YAML instruction.")
            self.step_lookup[step.id] = index

        self._current_observation: Optional[Observation] = None
        self._history: List[str] = []
        self._results: List[StepExecutionResult] = []
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(LangGraphInstructionState)
        builder.add_node("execute_current_step", self._execute_current_step)
        builder.add_edge(START, "execute_current_step")
        builder.add_conditional_edges(
            "execute_current_step",
            self._route_after_step,
            {
                "continue": "execute_current_step",
                "end": END,
            },
        )
        return builder.compile(debug=self.debug)

    def _default_next_step_id(self, step_index: int) -> Optional[str]:
        next_index = step_index + 1
        if next_index >= len(self.steps):
            return None
        return self.steps[next_index].id

    def _capture_observation(self) -> Observation:
        if self.observation_provider is not None:
            obs = self.observation_provider()
            if obs is None:
                raise ValueError("Observation provider returned None.")
            return obs
        return capture_desktop_observation()

    def _execute_current_step(
        self,
        state: LangGraphInstructionState,
    ) -> LangGraphInstructionState:
        step_counter = int(state.get("step_counter", 0)) + 1
        max_steps = int(state.get("max_steps", 100))
        if step_counter > max_steps:
            raise SafeWorkflowError(
                f"Instruction exceeded max_steps={max_steps}; possible loop detected."
            )

        current_step_id = state.get("current_step_id")
        if current_step_id is None:
            return {
                "current_step_id": None,
                "last_status": "completed",
                "last_error": None,
                "step_counter": step_counter,
                "max_steps": max_steps,
            }

        if current_step_id not in self.step_lookup:
            raise SafeWorkflowError(f"Unknown step id '{current_step_id}' in LangGraph state.")

        step_index = self.step_lookup[current_step_id]
        step = self.steps[step_index]
        if self._current_observation is None:
            self._current_observation = self._capture_observation()

        default_next_step_id = self._default_next_step_id(step_index)
        summary = execute_step(
            step,
            self._current_observation,
            step_index=step_index,
            template_threshold=self.template_threshold,
            observation_provider=self.observation_provider,
        )

        if not summary.can_execute:
            next_step_id = resolve_next_step_id(
                step,
                status="precondition_failed",
                verification=None,
                default_next_step_id=None,
            )
            self._history.append(f"precondition_failed:{step.id}")
            last_error = (
                summary.additionaal_info
                or f"Step '{step.id}' failed to meet its precondition."
            )
            if next_step_id is not None and next_step_id not in self.step_lookup:
                raise SafeWorkflowError(
                    f"Step '{step.id}' references unknown transition target '{next_step_id}'.",
                    step_id=step.id,
                )
            if next_step_id is None:
                return {
                    "current_step_id": None,
                    "last_status": "failed",
                    "last_error": last_error,
                    "step_counter": step_counter,
                    "max_steps": max_steps,
                }
            return {
                "current_step_id": next_step_id,
                "last_status": "branched_on_failure",
                "last_error": last_error,
                "step_counter": step_counter,
                "max_steps": max_steps,
            }

        try:
            result = summary.call_executable()
        except StepVerificationError as exc:
            next_step_id = resolve_next_step_id(
                step,
                status=exc.verification.status if exc.verification is not None else "verification_failed",
                verification=exc.verification,
                default_next_step_id=None,
            )
            self._current_observation = exc.observation or self._current_observation
            self._history.append(f"verification_failed:{step.id}")
            if next_step_id is not None and next_step_id not in self.step_lookup:
                raise SafeWorkflowError(
                    f"Step '{step.id}' references unknown transition target '{next_step_id}'.",
                    step_id=step.id,
                )
            if next_step_id is None:
                return {
                    "current_step_id": None,
                    "last_status": "failed",
                    "last_error": str(exc),
                    "step_counter": step_counter,
                    "max_steps": max_steps,
                }
            return {
                "current_step_id": next_step_id,
                "last_status": "branched_on_failure",
                "last_error": str(exc),
                "step_counter": step_counter,
                "max_steps": max_steps,
            }

        if not isinstance(result, StepExecutionResult):
            raise SafeWorkflowError(
                f"Step '{step.id}' did not return a step execution result.",
                step_id=step.id,
            )

        self._results.append(result)
        self._current_observation = result.after_observation
        self._history.append(f"success:{step.id}")
        next_step_id = resolve_next_step_id(
            step,
            status=result.verification.status,
            verification=result.verification,
            default_next_step_id=default_next_step_id,
        )
        if next_step_id is not None and next_step_id not in self.step_lookup:
            raise SafeWorkflowError(
                f"Step '{step.id}' references unknown transition target '{next_step_id}'.",
                step_id=step.id,
            )
        return {
            "current_step_id": next_step_id,
            "last_status": "completed" if next_step_id is None else "success",
            "last_error": None,
            "step_counter": step_counter,
            "max_steps": max_steps,
        }

    @staticmethod
    def _route_after_step(state: LangGraphInstructionState) -> str:
        if state.get("current_step_id") is None:
            return "end"
        return "continue"

    def run(
        self,
        *,
        obs: Optional[Observation] = None,
        max_steps: int = 100,
    ) -> LangGraphRunResult:
        self._history = []
        self._results = []
        self._current_observation = obs or self._capture_observation()

        initial_state: LangGraphInstructionState = {
            "current_step_id": self.steps[0].id if self.steps else None,
            "last_status": "initialized",
            "last_error": None,
            "step_counter": 0,
            "max_steps": max_steps,
        }
        final_state = self.graph.invoke(initial_state)
        return LangGraphRunResult(
            final_state=final_state,
            results=list(self._results),
            history=list(self._history),
            final_observation=self._current_observation,
        )


def run_safe_instruction_langgraph(
    instruction: YamlInstruction,
    *,
    obs: Optional[Observation] = None,
    template_threshold: float = 0.8,
    observation_provider: Optional[Callable[[], Observation]] = None,
    max_steps: int = 100,
    debug: bool = False,
) -> LangGraphRunResult:
    runner = LangGraphInstructionRunner(
        instruction,
        template_threshold=template_threshold,
        observation_provider=observation_provider,
        debug=debug,
    )
    return runner.run(obs=obs, max_steps=max_steps)


__all__ = [
    "LangGraphInstructionRunner",
    "LangGraphInstructionState",
    "LangGraphRunResult",
    "run_safe_instruction_langgraph",
]
