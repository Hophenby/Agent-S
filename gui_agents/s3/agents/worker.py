from functools import partial
import logging
import os
import textwrap
from typing import Dict, List, Tuple

from agents.grounding import ACI, LegacyACI
from core.module import BaseModule
from core.mllm import LMMAgent
from core.observation import Observation
from agents.execution_summary import ExecutionSummary
from agents.LegacyACIResult import LegacyACIResult
from instruction.yaml.yaml_instruction_auto_executor import execute_step
from instruction.yaml.yaml_instruction_parser import load_instruction
from instruction.yaml.yaml_instruction import YamlInstruction
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
    yaml_instruction: YamlInstruction
    yaml_step_index: int = 0
    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI = None,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        instruction_yaml_path: str = "",
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

        self.yaml_step_index = 0
        if self.instruction_yaml_path and os.path.isfile(self.instruction_yaml_path):
            self.yaml_instruction = load_instruction(self.instruction_yaml_path)

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

    def generate_next_action(self, instruction: str, obs: Observation) -> ExecutionSummary:
        """
        Predict the next action(s) based on the current observation.
        """

        self.grounding_agent.assign_screenshot(obs)
        # self.grounding_agent.set_task_instruction(instruction)

        step_plan = None

        if self.yaml_instruction and self.yaml_step_index < len(self.yaml_instruction.steps):
            step = self.yaml_instruction.steps[self.yaml_step_index]
            step_plan = execute_step(step, obs, self.yaml_step_index)
            if step_plan.can_execute:
                self.worker_history.append(step_plan.format_summary())
                self.yaml_step_index += 1
                self.turn_count += 1
                return step_plan

        plan = step_plan.format_summary() if step_plan else "No executable plan found in YAML instruction."

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
            instruction_reader_response=None,
        )
        self.turn_count += 1
        self.screenshot_inputs.append(obs.screenshot)
        self.flush_messages()
        return executor_info
