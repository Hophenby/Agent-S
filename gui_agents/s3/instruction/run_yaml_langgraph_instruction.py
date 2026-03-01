from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    _bootstrap_imports()

    from instruction.yaml.langgraph_instruction_runner import run_safe_instruction_langgraph
    from instruction.yaml.yaml_instruction_parser import load_instruction

    parser = argparse.ArgumentParser(description="Run a YAML safe instruction with LangGraph.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML instruction file.")
    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.8,
        help="Default template match threshold when a step does not override it.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of graph node executions before treating it as a loop.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable LangGraph debug mode.",
    )
    args = parser.parse_args()

    instruction = load_instruction(args.yaml_path)
    result = run_safe_instruction_langgraph(
        instruction,
        template_threshold=args.template_threshold,
        max_steps=args.max_steps,
        debug=args.debug,
    )
    print("Final state:", result.final_state)
    print("History:", result.history)
    print("Successful steps:", [item.step_id for item in result.results])


if __name__ == "__main__":
    main()
