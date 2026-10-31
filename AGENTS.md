# Repository Guidelines

## Project Structure & Module Organization

This repository fine-tunes DeepSeek-Coder for CUDA/HPC code using LoRA and Hugging Face tooling.

- `finetune_with_args.py` provides the training CLI; `dataset_loader.py` loads, filters, and formats datasets.
- `utils/` contains GPU diagnostics, monitoring, inference examples, and batch prompt generation.
- `prompts/<category>/<number>_<problem>/cuda` stores benchmark prompts grouped by problem type.
- `metrics/` holds evaluation CSVs. `generate_chart.py` and `generate_line_chart.py` produce plots in `graphs/`; `Graficos*` directories contain additional research figures.
- Consult `readme.md`, `QUICKSTART.md`, and `DATASET_LOADER_GUIDE.md` for usage and dataset details.

## Build, Test, and Development Commands

Run commands from the repository root in a virtual environment:

- `pip install -r requirements.txt` installs training and plotting dependencies.
- `python3 utils/check_compatibility.py` checks CUDA, GPU capabilities, and package compatibility.
- `python3 finetune_with_args.py --model_name deepseek-ai/deepseek-coder-1.3b-instruct --output_dir ./results --save_dir ./trained_model` starts basic fine-tuning. Training requires compatible GPU dependencies and access to model/dataset downloads.
- `python3 utils/monitor_gpu.py` monitors GPU resources.
- `python3 generate_chart.py` generates a bar chart; `python3 generate_line_chart.py` generates a metric-versus-k plot. Set each script's `INPUT_FILE` and `OUTPUT_FILE` first.

There is no separate build step.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase configuration constants. Preserve existing CLI flag names such as `--max_length`. Follow nearby docstrings and type annotations; existing documentation and logs include Portuguese. No formatter or linter is configured. Keep changes focused.

## Testing Guidelines

No automated test suite or coverage threshold is configured. Check changed Python files with `python3 -m py_compile <file.py>`. For dataset changes, validate filtering and formatted output with a small fixture; for charts, regenerate and visually inspect the affected plot. Record hardware, dependencies, commands, and results for training validation. If adding automated tests, use descriptive `test_*.py` filenames and document their runner.

## Commit & Pull Request Guidelines

History uses short, descriptive Portuguese and English commit subjects without a consistent prefix convention. Match that concise style. PRs should explain the change, link relevant issues, and report validation. Include before/after figures for chart changes and model, dataset, and training settings for experiment changes.

## Security & Configuration

Keep credentials, downloaded models, checkpoints, and local logs out of commits. `finetuned-models/` and `venv` are ignored; verify other output paths before staging files.
