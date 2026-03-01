import argparse
import datetime
import io
import logging
import os
import platform
import pyautogui
import signal
import sys
import time

from PIL import Image, ImageDraw, ImageFont

from agents.agent_s import AgentS3
from agents.grounding import LegacyACI, get_feedback_renderer
from core.observation import Observation
from instruction.yaml.yaml_instruction_auto_executor import SafeWorkflowError
from utils.common_utils import RUNTIME_LOG_PATH
from utils.local_env import LocalEnv

current_platform = platform.system().lower()

# Global flag to track pause state for debugging
paused = False


def get_char():
    """Get a single character from stdin without pressing Enter"""
    try:
        # Import termios and tty on Unix-like systems
        if platform.system() in ["Darwin", "Linux"]:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        else:
            # Windows fallback
            import msvcrt

            return msvcrt.getch().decode("utf-8", errors="ignore")
    except:
        return input()  # Fallback for non-terminal environments


def signal_handler(signum, frame):
    """Handle Ctrl+C signal for debugging during agent execution"""
    global paused

    if not paused:
        print("\n\n🔸 Agent-S Workflow Paused 🔸")
        print("=" * 50)
        print("Options:")
        print("  • Press Ctrl+C again to quit")
        print("  • Press Esc to resume workflow")
        print("=" * 50)

        paused = True

        while paused:
            try:
                print("\n[PAUSED] Waiting for input... ", end="", flush=True)
                char = get_char()

                if ord(char) == 3:  # Ctrl+C
                    print("\n\n🛑 Exiting Agent-S...")
                    sys.exit(0)
                elif ord(char) == 27:  # Esc
                    print("\n\n▶️  Resuming Agent-S workflow...")
                    paused = False
                    break
                else:
                    print(f"\n   Unknown command: '{char}' (ord: {ord(char)})")

            except KeyboardInterrupt:
                print("\n\n🛑 Exiting Agent-S...")
                sys.exit(0)
    else:
        # Already paused, second Ctrl+C means quit
        print("\n\n🛑 Exiting Agent-S...")
        sys.exit(0)


# Set up signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

platform_os = platform.system()


def show_permission_dialog(code: str, action_description: str):
    """Show a platform-specific permission dialog and return True if approved."""
    if platform.system() == "Darwin":
        result = os.system(
            f'osascript -e \'display dialog "Do you want to execute this action?\n\n{code} which will try to {action_description}" with title "Action Permission" buttons {{"Cancel", "OK"}} default button "OK" cancel button "Cancel"\''
        )
        return result == 0
    elif platform.system() == "Linux":
        result = os.system(
            f'zenity --question --title="Action Permission" --text="Do you want to execute this action?\n\n{code}" --width=400 --height=200'
        )
        return result == 0
    return False


def scale_screen_dimensions(width: int, height: int, max_dim_size: int):
    scale_factor = min(max_dim_size / width, max_dim_size / height, 1)
    safe_width = int(width * scale_factor)
    safe_height = int(height * scale_factor)
    return safe_width, safe_height

def screenshot(scaled_width: int, scaled_height: int) ->bytes:
        screenshot = pyautogui.screenshot()

        # Resize for target dimensions
        screenshot = screenshot.resize((scaled_width, scaled_height), Image.LANCZOS)

        # Flatten alpha to white background if needed, then ensure RGB
        if screenshot.mode == "RGBA":
            bg = Image.new("RGB", screenshot.size, (255, 255, 255))
            bg.paste(screenshot, mask=screenshot.split()[3])
            screenshot = bg
        elif screenshot.mode != "RGB":
            screenshot = screenshot.convert("RGB")

        # Reduce color palette to significantly cut PNG size while keeping readability.
        # 128 colors is a good balance; reduce further (e.g., 64) if more compression is needed.
        try:
            screenshot = screenshot.quantize(colors=128, method=Image.MEDIANCUT)
        except Exception:
            # Fallback: leave image as-is if quantization fails
            pass

        # Save the screenshot to a BytesIO object
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")

        # Get the byte value of the screenshot
        return buffered.getvalue()



def run_agent(agent: AgentS3, instruction: str, scaled_width: int, scaled_height: int):
    """Run the agent for up to 15 steps, feeding screenshots and handling feedback images.

    This replaces the live screenshot with an annotated feedback image when the
    grounding agent provides one. It keeps the previous behavior otherwise.
    
    Args:
        agent: The AgentS3 instance
        instruction: The task instruction
        scaled_width: Screenshot width
        scaled_height: Screenshot height
        bbox_annotation_path: Optional path to bounding box annotation JSON file
    """
    global paused
    obs = Observation()
    traj = "Task:\n" + instruction
    subtask_traj = ""
    fb_note = ""

    bbox_annotation_data = None
    enhanced_instruction = instruction

    for step in range(15):
        # Check if we're in paused state and wait
        while paused:
            time.sleep(0.1)

        # Get screen shot using pyautogui (or use previously-rendered annotated screenshot)
        # screenshot_bytes = screenshot(scaled_width, scaled_height) if next_screenshot_bytes is None else next_screenshot_bytes

        if callable(get_feedback_renderer()):
            try:
                screenshot_bytes = get_feedback_renderer()(screenshot(scaled_width, scaled_height))
                print("🖼️  Using annotated feedback screenshot from grounding agent.")
            except Exception:
                screenshot_bytes = screenshot(scaled_width, scaled_height)
        else:
            print("🖼️  Using regular screenshot (no feedback annotation).")
            screenshot_bytes = screenshot(scaled_width, scaled_height)

        print(f"\n📸 Captured screenshot of size: {len(screenshot_bytes) / 1024:.2f} KB")

        # Prepare observation
        obs.screenshot = screenshot_bytes
        original_byteio = io.BytesIO()
        pyautogui.screenshot().save(original_byteio, format="PNG")
        obs.original_screenshot = original_byteio.getvalue()
        
        # Add bounding box annotations to observation if available
        if bbox_annotation_data:
            obs["ui_annotations"] = bbox_annotation_data
            obs["ui_element_names"] = [
                ann.get("name") for ann in bbox_annotation_data.get("annotations", [])
            ]
        
        with open(
            os.path.join(RUNTIME_LOG_PATH, f"step_{step + 1}_screenshot.png"), "wb"
        ) as f:
            f.write(screenshot_bytes)
        
        print(f"📝 Current Trajectory:\n{traj}\n")

        # Check again for pause state before prediction
        while paused:
            time.sleep(0.1)

        print(f"\n🔄 Step {step + 1}/15: Getting next action from agent...")

        # Get next action code from the agent with enhanced instruction
        summary = agent.predict(instruction=enhanced_instruction, observation=obs)

        if summary.executable is None:
            print("⚠️  No code returned by agent, stopping execution.")
            break

        # Extract the first action and normalize to an executable string.
        exec_str = summary.exec_str

        print(f"📝 Agent returned action code:\n{exec_str}\n")
        if exec_str is None:
            print("⚠️  No code returned by agent, stopping execution.")
            break

        if ("done" in exec_str.lower() or "fail" in exec_str.lower()):
            if platform.system() == "Darwin":
                os.system(
                    f'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                )
            elif platform.system() == "Linux":
                os.system(
                    f'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                )

            break

        if "next" in exec_str.lower():
            continue

        if "wait" in exec_str.lower():
            print("⏳ Agent requested wait...")
            time.sleep(5)
            continue

        # Otherwise execute the code
        time.sleep(1.0)
        print("EXECUTING CODE:", exec_str)

        # Check for pause state before execution
        while paused:
            time.sleep(0.1)

        # Ask for permission before executing
        try:
            result = summary.call_executable()
        except SafeWorkflowError as exc:
            print(f"⚠️  Safe runner stopped: {exc}")
            break
        if getattr(getattr(agent, "executor", None), "yaml_runtime", None) == "langgraph":
            last_error = getattr(agent.executor, "yaml_failure_message", None)
            if last_error:
                print(f"⚠️  LangGraph safe runner stopped: {last_error}")
            else:
                print("✅ LangGraph safe runner completed.")
            break
        time.sleep(1.0)

        # Update task and subtask trajectories
        if summary.additionaal_info is not None and summary.plan is not None:
            traj += (
                "\n\nReflection:\n"
                + str(summary.additionaal_info)
                + "\n\n----------------------\n\nPlan:\n"
                + summary.plan
            )


def main():
    parser = argparse.ArgumentParser(description="Run AgentS3 with specified model.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Specify the provider to use (e.g., openai, anthropic, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-2025-08-07",
        help="Specify the model to use (e.g., gpt-5-2025-08-07)",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL of the main generation model API.",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default="",
        help="The API key of the main generation model.",
    )
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=None,
        help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)",
    )

    # Grounding model config: Self-hosted endpoint based (required)
    parser.add_argument(
        "--ground_provider",
        type=str,
        required=True,
        help="The provider for the grounding model",
    )
    parser.add_argument(
        "--ground_url",
        type=str,
        required=True,
        help="The URL of the grounding model",
    )
    parser.add_argument(
        "--ground_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )
    parser.add_argument(
        "--ground_model",
        type=str,
        required=True,
        help="The model name for the grounding model",
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        required=True,
        help="Width of screenshot image after processor rescaling",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        required=True,
        help="Height of screenshot image after processor rescaling",
    )

    # AgentS3 specific arguments
    parser.add_argument(
        "--max_trajectory_length",
        type=int,
        default=8,
        help="Maximum number of image turns to keep in trajectory",
    )
    parser.add_argument(
        "--enable_reflection",
        action="store_true",
        default=True,
        help="Enable reflection agent to assist the worker agent",
    )
    parser.add_argument(
        "--enable_local_env",
        action="store_true",
        default=False,
        help="Enable local coding environment for code execution (WARNING: Executes arbitrary code locally)",
    )
    parser.add_argument(
        "--instruction_yaml_path",
        "--instruction_markdown_path",
        dest="instruction_yaml_path",
        type=str,
        default="",
        help="Path to the YAML instruction file for safe-runner mode.",
    )
    parser.add_argument(
        "--yaml_runtime",
        type=str,
        default="native",
        choices=["native", "langgraph"],
        help="YAML safe-runner runtime: native stepwise executor or LangGraph workflow runtime.",
    )

    args = parser.parse_args()

    # Re-scales screenshot size to ensure it fits in UI-TARS context limit
    screen_width, screen_height = pyautogui.size()
    # scaled_width, scaled_height = scale_screen_dimensions(
    #     screen_width, screen_height, max_dim_size=2400
    # )
    scaled_width, scaled_height = args.grounding_width, args.grounding_height 
    # Load the general engine params
    engine_params = {
        "engine_type": args.provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
        "temperature": getattr(args, "model_temperature", None),
    }

    # Load the grounding engine from a custom endpoint
    # engine_params_for_grounding = {
    #     "engine_type": args.ground_provider,
    #     "model": args.ground_model,
    #     "base_url": args.ground_url,
    #     "api_key": args.ground_api_key,
    #     "grounding_width": args.grounding_width,
    #     "grounding_height": args.grounding_height,
    # }

    # Initialize environment based on user preference
    local_env = None
    if args.enable_local_env:
        print(
            "⚠️  WARNING: Local coding environment enabled. This will execute arbitrary code locally!"
        )
        local_env = LocalEnv()

    grounding_agent = LegacyACI(
        # env=local_env,
        # platform=current_platform,
        # engine_params_for_generation=engine_params,
        # engine_params_for_grounding=engine_params_for_grounding,
        width=scaled_width,
        height=screen_height,
    )

    agent = AgentS3(
        engine_params,
        grounding_agent,
        platform=current_platform,
        max_trajectory_length=args.max_trajectory_length,
        enable_reflection=args.enable_reflection,
        instruction_yaml_path=args.instruction_yaml_path,
        yaml_runtime=args.yaml_runtime,
    )

    while True:
        query = input("Query: ")
        print("Delaying for 2 seconds before starting the agent...")
        time.sleep(2.0)

        agent.reset()

        # Run the agent on your own device
        run_agent(agent, query, args.grounding_width, args.grounding_height)

        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
