if exist D:\deepresearch\open_deep_research\.env (
    for /f "usebackq delims=" %%i in ("D:\deepresearch\open_deep_research\.env") do set %%i
)
set HTTPS_PROXY=http://localhost:10808
set HTTP_PROXY=http://localhost:10808
set HF_ENDPOINT_URL=https://router.huggingface.co/v1

python D:\agents\Agent-S\gui_agents\s3\cli_app.py ^
    --provider openai ^
    --model gpt-5-2025-08-07 ^
    --ground_provider huggingface ^
    --ground_url https://router.huggingface.co/v1 ^
    --ground_model ui-tars-1.5-7b ^
    --grounding_width 2560 ^
    --grounding_height 1440 ^
    --model_temperature 1 ^
    --enable_local_env ^
    --instruction_markdown_path D:\agents\Agent-S\gui_agents\s3\agents\example_instruction.md
