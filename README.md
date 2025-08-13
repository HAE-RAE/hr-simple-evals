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

### KoSimpleEval 통합 데이터셋 (권장)
모든 데이터셋이 `HAERAE-HUB/KoSimpleEval`에 통합되어 있어 일관된 형식으로 사용할 수 있습니다:

- `ArenaHard` - Arena Hard 벤치마크 (500 samples)
- `ClinicalQA` - 임상 질의응답 (1045 samples)
- `HRB1_0` - HAERAE Benchmark 1.0 (1538 samples)
- `KMMLU_Redux` - 한국어 MMLU 축약판 (2742 samples)
- `KMMLU` - 한국어 MMLU 전체 (35030 samples)
- `KMMLU-Pro` - KMMLU 고급 버전 (2822 samples)
- `KMMLU-HARD` - KMMLU 어려운 버전 (4104 samples)
- `KorMedLawQA` - 한국 의료법 질의응답 (13388 samples)
- `CLIcK` - 한국 상식 추론 (1995 samples)
- `MCLM` - 수학 문제 (129 samples)
- `GPQA` - 과학 질의응답 (198 samples)
- `KoBALT-700` - 한국어 균형 언어 테스트 (700 samples)
- `AIME2024` - 2024년 AIME 수학 문제 (30 samples)
- `AIME2025` - 2025년 AIME 수학 문제 (30 samples)
- `KSM` - 한국어 수학 문제 (1428 samples)

### 하위 호환성
기존 외부 데이터셋 이름들도 자동으로 KoSimpleEval로 리다이렉트됩니다:
- `gpqa-diamond` → `GPQA`
- `aime_2024` → `AIME2024`  
- `aime_2025` → `AIME2025`

Replace `<model-id-or-path>` with the Hugging Face model ID or a local checkpoint.

The script will generate responses using the specified model and evaluate them according to the dataset configuration defined in `dataset_configs.py`.

## 사용 예시

### KoSimpleEval 통합 데이터셋 사용 (권장)
모든 데이터셋이 통합되어 있어 간단하게 사용할 수 있습니다:

```bash
# AIME 2025 수학 문제
python evaluation.py \
  --model <model-id-or-path> \
  --dataset AIME2025 \
  --temperature 0.0 \
  --max_tokens 1024

# AIME 2024 수학 문제  
python evaluation.py \
  --model <model-id-or-path> \
  --dataset AIME2024 \
  --temperature 0.0 \
  --max_tokens 1024

# GPQA 과학 질의응답
python evaluation.py \
  --model <model-id-or-path> \
  --dataset GPQA \
  --temperature 0.0 \
  --max_tokens 1024

# 한국어 수학 문제 (KSM)
python evaluation.py \
  --model <model-id-or-path> \
  --dataset KSM \
  --temperature 0.0 \
  --max_tokens 1024
```

### 하위 호환성
기존 명령어도 자동으로 KoSimpleEval로 리다이렉트됩니다:

```bash
# 이전 방식 (여전히 작동함)
python evaluation.py \
  --model <model-id-or-path> \
  --dataset gpqa-diamond \
  --temperature 0.0 \
  --max_tokens 1024
```

이 명령어는 자동으로 `HAERAE-HUB/KoSimpleEval`의 `GPQA` config를 사용합니다.
