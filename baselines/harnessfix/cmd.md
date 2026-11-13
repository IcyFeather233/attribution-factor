```bash
source .env && .venv/bin/python3 run_pipeline_gaia.py \
    --model openai/gemini-2.0-flash \
    --analysis-model openai/claude-sonnet-4-5-20250929 \
    --max-iterations 2 \
    --workers 2 \
    --concurrency 4 \
    --start-version 1 
```