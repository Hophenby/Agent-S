from functools import partial
import logging
import os
import textwrap
from typing import Dict, List, Optional, Tuple

from agents.grounding import ACI, LegacyACI
from core.module import BaseModule
from core.mllm import LMMAgent
from core.observation import Observation
from agents.execution_summary import ExecutionSummary
from agents.LegacyACIResult import LegacyACIResult
from instruction.yaml.yaml_instruction_auto_executor import (
    StepVerificationError,
    execute_step,
    resolve_next_step_id,
)
from instruction.yaml.yaml_instruction_parser import load_instruction
from instruction.yaml.yaml_instruction import Step, YamlInstruction
from instruction.instruction_reader import InstructionReader, process_generation_result
from memory.procedural_memory import PROCEDURAL_MEMORY
from utils.common_utils import (
    call_llm_safe,
    call_llm_formatted,
    parse_code_from_string,
    split_thinking_response,
    create_pyautogui_code,
)
from utils.formatters import (
    SINGLE_ACTION_FORMATTER,
    CODE_VALID_FORMATTER,
)

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    grounding_agent: LegacyACI
    generator_agent: LMMAgent
    reflection_agent: LMMAgent
    instruction_agent: InstructionReader
    yaml_instruction: Optional[YamlInstruction] = None
    yaml_step_index: Optional[int] = 0
    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI = None,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        instruction_yaml_path: str = "",
        yaml_runtime: str = "native",
    ):
        """
        Worker receives the main task and generates actions, without the need of hierarchical planning
        Args:
            worker_engine_params: Dict
                Parameters for the worker agent
            grounding_agent: Agent
                The grounding agent to use
            platform: str
                OS platform the agent runs on (darwin, linux, windows)
            max_trajectory_length: int
                The amount of images turns to keep
            enable_reflection: bool
                Whether to enable reflection
        """
        super().__init__(worker_engine_params, platform)

        self.temperature = worker_engine_params.get("temperature", 0.0)
        self.use_thinking = worker_engine_params.get("model", "") in [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-5-20250929",
        ]
        # If no grounding_agent provided, default to LegacyACI
        if grounding_agent is None:
            self.grounding_agent: LegacyACI = LegacyACI()
        else:
            self.grounding_agent: LegacyACI = grounding_agent
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.instruction_yaml_path = instruction_yaml_path
        self.yaml_runtime = yaml_runtime

        self.reset()

    def reset(self):
        if self.platform != "linux":
            skipped_actions = ["set_cell_values"]
        else:
            skipped_actions = []

        # Hide code agent action entirely if no env/controller is available
        if not getattr(self.grounding_agent, "env", None) or not getattr(
            getattr(self.grounding_agent, "env", None), "controller", None
        ):
            skipped_actions.append("call_code_agent")

        self.yaml_instruction = None
        self.yaml_step_index = 0
        self.yaml_step_lookup: Dict[str, int] = {}
        self.yaml_failure_message: Optional[str] = None
        self.yaml_graph_finished = False
        if self.instruction_yaml_path and os.path.isfile(self.instruction_yaml_path):
            self.yaml_instruction = load_instruction(self.instruction_yaml_path)
            self.yaml_step_lookup = {
                step.id: index for index, step in enumerate(self.yaml_instruction.steps)
            }

        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )#TODO: 这个还没修改

        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []

    def flush_messages(self):
        """Flush messages based on the model's context limits.

        This method ensures that the agent's message history does not exceed the maximum trajectory length.

        Side Effects:
            - Modifies the messages of generator, reflection, and bon_judge agents to fit within the context limits.
        """
        engine_type = self.engine_params.get("engine_type", "")

        # Flush strategy for long-context models: keep all text, only keep latest images
        if engine_type in ["anthropic", "openai", "gemini"]:
            max_images = self.max_trajectory_length
            for agent in [self.reflection_agent]:
                if agent is None:
                    continue
                # keep latest k images
                img_count = 0
                for i in range(len(agent.messages) - 1, -1, -1):
                    for j in range(len(agent.messages[i]["content"])):
                        if "image" in agent.messages[i]["content"][j].get("type", ""):
                            img_count += 1
                            if img_count > max_images:
                                del agent.messages[i]["content"][j]

        # Flush strategy for non-long-context models: drop full turns
        else:
            # reflector msgs are all [(user text, user image)], so 1 per round
            if len(self.reflection_agent.messages) > self.max_trajectory_length + 1:
                self.reflection_agent.messages.pop(1)

    def _generate_reflection(self, instruction: str, obs: Observation) -> Tuple[str, str]:
        """
        Generate a reflection based on the current observation and instruction.

        Args:
            instruction (str): The task instruction.
            obs (Dict): The current observation containing the screenshot.

        Returns:
            Optional[str, str]: The generated reflection text and thoughts, if any (turn_count > 0).

        Side Effects:
            - Updates reflection agent's history
            - Generates reflection response with API call
        """
        reflection = None
        reflection_thoughts = None
        if self.enable_reflection:
            # Load the initial message
            if self.turn_count == 0:
                text_content = textwrap.dedent(
                    f"""
                    Task Description: {instruction}
                    Current Trajectory below:
                    """
                )
                updated_sys_prompt = (
                    self.reflection_agent.system_prompt + "\n" + text_content
                )
                self.reflection_agent.add_system_prompt(updated_sys_prompt)
                self.reflection_agent.add_message(
                    text_content="The initial screen is provided. No action has been taken yet.",
                    image_content=obs.screenshot,
                    role="user",
                )
            # Load the latest action
            else:
                self.reflection_agent.add_message(
                    text_content=self.worker_history[-1],
                    image_content=obs.screenshot,
                    role="user",
                )
                full_reflection = call_llm_safe(
                    self.reflection_agent,
                    temperature=self.temperature,
                    use_thinking=self.use_thinking,
                )
                reflection, reflection_thoughts = split_thinking_response(
                    full_reflection
                )
                self.reflections.append(reflection)
                logger.info("REFLECTION THOUGHTS: %s", reflection_thoughts)
                logger.info("REFLECTION: %s", reflection)
        return reflection, reflection_thoughts

    def _resolve_yaml_next_step_index(
        self,
        next_step_id: Optional[str],
        *,
        step: Step,
        branch_name: str,
        fallback_index: Optional[int],
    ) -> Optional[int]:
        if next_step_id is None:
            return fallback_index
        if next_step_id not in self.yaml_step_lookup:
            raise ValueError(
                f"Step '{step.id}' references unknown {branch_name} target '{next_step_id}'."
            )
        return self.yaml_step_lookup[next_step_id]

    def _wrap_yaml_execution_summary(
        self,
        step: Step,
        step_index: int,
        summary: ExecutionSummary,
    ) -> ExecutionSummary:
        original_executable = summary.executable

        def wrapped_executable():
            default_next_step_id = (
                self.yaml_instruction.steps[step_index + 1].id
                if self.yaml_instruction is not None
                and step_index + 1 < len(self.yaml_instruction.steps)
                else None
            )

            try:
                if callable(original_executable):
                    result = original_executable()
                elif isinstance(original_executable, str):
                    result = exec(original_executable)
                else:
                    result = None
            except StepVerificationError as exc:
                next_step_id = resolve_next_step_id(
                    step,
                    status=exc.verification.status if exc.verification is not None else "verification_failed",
                    verification=exc.verification,
                    default_next_step_id=None,
                )
                failure_index = self._resolve_yaml_next_step_index(
                    next_step_id,
                    step=step,
                    branch_name="transition",
                    fallback_index=None,
                )
                if failure_index is None:
                    self.yaml_failure_message = str(exc)
                    self.yaml_step_index = None
                    raise
                self.yaml_step_index = failure_index
                return exc

            next_step_id = resolve_next_step_id(
                step,
                status=getattr(getattr(result, "verification", None), "status", "success"),
                verification=getattr(result, "verification", None),
                default_next_step_id=default_next_step_id,
            )
            self.yaml_step_index = self._resolve_yaml_next_step_index(
                next_step_id,
                step=step,
                branch_name="transition",
                fallback_index=None,
            )
            return result

        return ExecutionSummary(
            plan=summary.plan,
            plan_action=summary.plan_action,
            executable=wrapped_executable,
            additionaal_info=summary.additionaal_info,
            reflection_thoughts=summary.reflection_thoughts,
        )

    def _build_langgraph_execution_summary(self, obs: Observation) -> ExecutionSummary:
        def wrapped_executable():
            try:
                from instruction.yaml.langgraph_instruction_runner import run_safe_instruction_langgraph
            except ImportError as exc:
                self.yaml_failure_message = str(exc)
                self.yaml_graph_finished = True
                raise

            result = run_safe_instruction_langgraph(
                self.yaml_instruction,
                obs=obs,
            )
            self.yaml_graph_finished = True
            if result.final_state.get("last_status") == "failed":
                self.yaml_failure_message = result.final_state.get("last_error") or "LangGraph safe runner failed."
            return result

        return ExecutionSummary(
            plan="Run YAML safe runner with LangGraph.",
            plan_action="Execute the compiled conditional workflow graph.",
            executable=wrapped_executable,
            additionaal_info="yaml_runtime=langgraph",
        )

    def generate_next_action(self, instruction: str, obs: Observation) -> ExecutionSummary:
        """
        Predict the next action(s) based on the current observation.
        """

        self.grounding_agent.assign_screenshot(obs)
        # self.grounding_agent.set_task_instruction(instruction)

        if self.yaml_instruction is not None:
            if self.yaml_runtime == "langgraph":
                if self.yaml_failure_message is not None:
                    return ExecutionSummary(
                        plan="YAML LangGraph runner stopped.",
                        plan_action="",
                        executable=None,
                        additionaal_info=self.yaml_failure_message,
                    )
                if self.yaml_graph_finished:
                    return ExecutionSummary(
                        plan="YAML LangGraph runner completed.",
                        plan_action="",
                        executable=None,
                        additionaal_info="The LangGraph workflow has finished.",
                    )

                summary = self._build_langgraph_execution_summary(obs)
                self.worker_history.append(summary.format_summary())
                self.turn_count += 1
                return summary

            if self.yaml_failure_message is not None:
                return ExecutionSummary(
                    plan="YAML safe runner stopped.",
                    plan_action="",
                    executable=None,
                    additionaal_info=self.yaml_failure_message,
                )

            steps = self.yaml_instruction.steps
            if self.yaml_step_index is None or self.yaml_step_index >= len(steps):
                return ExecutionSummary(
                    plan="YAML safe runner completed.",
                    plan_action="",
                    executable=None,
                    additionaal_info="All YAML steps have finished.",
                )

            branch_hops = 0
            while self.yaml_step_index is not None and self.yaml_step_index < len(steps):
                if branch_hops > len(steps) + 5:
                    self.yaml_failure_message = "YAML safe runner exceeded branch resolution limit."
                    return ExecutionSummary(
                        plan="YAML safe runner stopped.",
                        plan_action="",
                        executable=None,
                        additionaal_info=self.yaml_failure_message,
                    )

                step = steps[self.yaml_step_index]
                step_plan = execute_step(step, obs, self.yaml_step_index)
                if step_plan.can_execute:
                    wrapped_plan = self._wrap_yaml_execution_summary(
                        step,
                        self.yaml_step_index,
                        step_plan,
                    )
                    self.worker_history.append(wrapped_plan.format_summary())
                    self.turn_count += 1
                    return wrapped_plan

                next_step_id = resolve_next_step_id(
                    step,
                    status="precondition_failed",
                    verification=None,
                    default_next_step_id=None,
                )
                failure_index = self._resolve_yaml_next_step_index(
                    next_step_id,
                    step=step,
                    branch_name="transition",
                    fallback_index=None,
                )
                if failure_index is None:
                    self.yaml_failure_message = (
                        step_plan.additionaal_info
                        or f"Step '{step.id}' failed to meet its precondition."
                    )
                    self.yaml_step_index = None
                    return ExecutionSummary(
                        plan=step_plan.plan,
                        plan_action=step_plan.plan_action,
                        executable=None,
                        additionaal_info=self.yaml_failure_message,
                    )

                self.yaml_step_index = failure_index
                branch_hops += 1

            return ExecutionSummary(
                plan="YAML safe runner completed.",
                plan_action="",
                executable=None,
                additionaal_info="All YAML steps have finished.",
            )

        plan = "No executable plan found in YAML instruction."

        # Get the per-step reflection
        reflection, reflection_thoughts = self._generate_reflection(instruction, obs)

        self.worker_history.append(plan)
        logger.info("PLAN:\n %s", plan)

        executor_info = ExecutionSummary(
            plan=plan,
            plan_action="",
            executable=lambda: None,
            additionaal_info=reflection,
            reflection_thoughts=reflection_thoughts,
        )
        self.turn_count += 1
        self.screenshot_inputs.append(obs.screenshot)
        self.flush_messages()
        return executor_info
