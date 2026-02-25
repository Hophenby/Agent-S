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
    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI = None,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        instruction_markdown_path: str = "",
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
        self.instruction_markdown_path = instruction_markdown_path
        self.has_instruction_markdown = False

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

        sys_prompt = PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
            type(self.grounding_agent), 
            skipped_actions=skipped_actions, 
            guidelines=PROCEDURAL_MEMORY.TASK_DESCRIPTION_GUIDELINES, 
            formatting_instructions=PROCEDURAL_MEMORY.RESPONSE_FORMAT_PROMPT
        ).replace("CURRENT_OS", self.platform)

        self.generator_agent = self._create_agent(sys_prompt)
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )
        if (os.path.isfile(self.instruction_markdown_path)):
            self.instruction_agent = InstructionReader(
                llm_client=self._create_agent(),
                temperature=self.temperature,
                use_thinking=self.use_thinking,
            )
            self.instruction_agent.load_instruction_from_markdown(
                self.instruction_markdown_path
            )
            self.has_instruction_markdown = True
        else:
            self.instruction_agent = InstructionReader(
                llm_client=self._create_agent(),
                temperature=self.temperature,
                use_thinking=self.use_thinking,
            )
            self.has_instruction_markdown = False

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
            for agent in [self.generator_agent, self.reflection_agent]:
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
            # generator msgs are alternating [user, assistant], so 2 per round
            if len(self.generator_agent.messages) > 2 * self.max_trajectory_length + 1:
                self.generator_agent.messages.pop(1)
                self.generator_agent.messages.pop(1)
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

        generator_message = (
            "The screen after the last action is provided. It has marked the location of the last action taken."
            if self.turn_count > 0
            else "The initial screen is provided. No action has been taken yet."
        )

        # Load the task into the system prompt
        if self.turn_count == 0:
            prompt_with_instructions = self.generator_agent.system_prompt.replace(
                "TASK_DESCRIPTION", instruction
            )
            self.generator_agent.add_system_prompt(prompt_with_instructions)

        # Get the per-step reflection
        reflection, reflection_thoughts = self._generate_reflection(instruction, obs)
        if reflection:
            generator_message += f"REFLECTION: You may use this reflection on the previous action and overall trajectory:\n{reflection}\n"

        # Get the grounding agent's knowledge base buffer
        generator_message += (
            f"\nCurrent Text Buffer = [{','.join(self.grounding_agent.notes)}]\n"
        )

        if self.has_instruction_markdown:
            instruction_reader_response, executed_codes = self.instruction_agent.run_generation(
                observation=obs,
                instruction=instruction,
                generator_message=generator_message,
            )
            print("instruction_reader: Executed Codes:", executed_codes)

            instruction_reader_response = process_generation_result(obs, instruction_reader_response)
            if (instruction_reader_response.get("match_box") is not None):
                generator_message += f"用户已经在软件的使用说明书中找到了下一步需要操作的区域，对应的区域位置为: {instruction_reader_response.get('match_box')}"

        # Finalize the generator message
        self.generator_agent.add_message(
            generator_message, image_content=obs.screenshot, role="user"
        )

        # Generate the plan and next action
        format_checkers = [
            SINGLE_ACTION_FORMATTER,
            partial(CODE_VALID_FORMATTER, self.grounding_agent, obs),
        ]
        plan = call_llm_formatted(
            self.generator_agent,
            format_checkers,
            temperature=self.temperature,
            use_thinking=self.use_thinking,
        )
        self.worker_history.append(plan)
        self.generator_agent.add_message(plan, role="assistant")
        logger.info("PLAN:\n %s", plan)

        # Extract the next action from the plan
        plan_code = parse_code_from_string(plan)
        try:
            assert plan_code, "Plan code should not be empty"
            exec_code = create_pyautogui_code(self.grounding_agent, plan_code, obs)
            # If the grounding agent produced structured feedback (LegacyACI),
            # create_pyautogui_code sets agent.last_action_feedback to that LegacyACIResult.
            # Feed the annotated feedback image + annotation back into the
            # generator_agent context so the next planning step can use it.
            try:
                fb = getattr(self.grounding_agent, "last_action_feedback", None)
                if fb and isinstance(fb, LegacyACIResult):
                    annotation = fb.annotation
                    img_b64 = fb.feedback_image_base64
                    # Add as a user message (environment feedback) with image
                    self.generator_agent.add_message(
                        text_content=(f"ENV FEEDBACK: {annotation}" if annotation else "ENV FEEDBACK"),
                        image_content=img_b64,
                        role="user",
                    )
                    # Also add to reflection agent if available so reflection can see it
                    if getattr(self, "reflection_agent", None):
                        try:
                            self.reflection_agent.add_message(
                                text_content=(f"ENV FEEDBACK: {annotation}" if annotation else "ENV FEEDBACK"),
                                image_content=img_b64,
                                role="user",
                            )
                        except Exception:
                            pass
            except Exception:
                # best-effort: don't break main flow on feedback attachment failure
                pass
        except Exception as e:
            logger.error(
                f"Could not evaluate the following plan code:\n{plan_code}\nError: {e}"
            )
            exec_code = None

        executor_info = ExecutionSummary(
            plan=plan,
            plan_code=plan_code,
            executable=exec_code,
            reflection=reflection,
            reflection_thoughts=reflection_thoughts,
            instruction_reader_response=instruction_reader_response if self.has_instruction_markdown else None,
        )
        self.turn_count += 1
        self.screenshot_inputs.append(obs.screenshot)
        self.flush_messages()
        return executor_info
