# hr-simple-evals

This repository contains simple evaluation utilities for Korean language models.

## Running evaluations

Use `evaluation.py` to run a model on a dataset from the [HAERAE-HUB/KoSimpleEval](https://huggingface.co/datasets/HAERAE-HUB/KoSimpleEval) collection. The basic command looks like:

```bash
python evaluation.py \
  --model <model-id-or-path> \
  --dataset <subset-name> \
  --dataset_hub_id HAERAE-HUB/KoSimpleEval \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_tokens 1024
```

Supported subset names include:

- `ArenaHard`
- `ClinicalQA`
- `HRB1_0`
- `KMMLU_Redux`
- `MCLM`
- `gpqa-diamond` (You Should use Idavidrein/gpqa - gpqa-diamond subset for evaluation)
- `KSM` (You Should use HAERAE-HUB/HRM8K - KSM subset for evaluation)
- `AIME2024` (You Should use HuggingFaceH4/aime_2024 train subset for evaluation)
- `AIME2025` (You Should use yentinglin/aime_2025 default subset for evaluation)

Replace `<model-id-or-path>` with the Hugging Face model ID or a local checkpoint.

The script will generate responses using the specified model and evaluate them according to the dataset configuration defined in `dataset_configs.py`.

## Using AIME datasets

You can evaluate models on the AIME (American Invitational Mathematics Examination) datasets:

### AIME 2025
```bash
python evaluation.py \
  --model <model-id-or-path> \
  --dataset_hub_id yentinglin/aime_2025 \
  --split default \
  --temperature 0.0 \
  --max_tokens 1024
```

### AIME 2024
```bash
python evaluation.py \
  --model <model-id-or-path> \
  --dataset_hub_id HuggingFaceH4/aime_2024 \
  --split train \
  --temperature 0.0 \
  --max_tokens 1024
```

These datasets contain challenging mathematics problems that test a model's mathematical reasoning capabilities.
