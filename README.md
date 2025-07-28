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

Replace `<model-id-or-path>` with the Hugging Face model ID or a local checkpoint.

The script will generate responses using the specified model and evaluate them according to the dataset configuration defined in `dataset_configs.py`.
