#!/usr/bin/env python
import argparse
import os
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
from dataset_configs import get_config

# LiteLLM을 위한 API 키 설정이 필요하면 여기에서
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

def parse_args():
    """스크립트 실행을 위한 인자들을 파싱합니다."""
    p = argparse.ArgumentParser(description="Run a vLLM model on a dataset and save its responses.")
    p.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    p.add_argument("--dataset", required=True, help=f"Dataset name. Supported: {list(get_config('').keys())}")
    p.add_argument("--dataset_hub_id", default='HAERAE-HUB/KoSimpleEval', help="Hugging Face Hub ID for the dataset collection")
    p.add_argument("--split", default="test", help="Dataset split (default: test)")
    p.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for deterministic output)")
    p.add_argument("--top_p", type=float, default=1.0, help="Top-p / nucleus sampling (1.0 to disable)")
    p.add_argument("--output", default=None, help="Output CSV path (auto-generated if omitted)")
    return p.parse_args()

def main() -> None:
    """메인 평가 파이프라인을 실행합니다."""
    args = parse_args()

    # 1. 설정 가져오기
    try:
        config = get_config(args.dataset)
        prompt_maker = config['prompt_maker']
        evaluator = config['evaluator']
    except ValueError as e:
        print(f"오류: {e}")
        return
    
    print(f"🚀 모델 로딩 중: {args.model}")
    llm = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
    tokenizer = llm.get_tokenizer()

    print(f"📚 데이터셋 로딩 중: {args.dataset_hub_id} - {args.dataset}")
    df = load_dataset(args.dataset_hub_id, args.dataset, split=args.split).to_pandas()

    print("✍️  프롬프트를 생성하고 있습니다...")
    tqdm.pandas(desc="프롬프트 생성")
    prompts = df.progress_apply(lambda row: prompt_maker(row, tokenizer), axis=1).tolist()

    print(f"🧠 모델 응답을 생성하고 있습니다 (총 {len(prompts)}개)...")
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling_params)
    df["response"] = [o.outputs[0].text.strip() for o in outputs]

    result_df = evaluator(df, args)

    if args.output is None:
        safe_model = args.model.replace("/", "_")
        safe_data = args.dataset.replace("/", "_")
        args.output = f"results/{safe_data}-{safe_model}.csv"
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"✅💾 {len(result_df)}개 행의 결과 저장 완료 ➜ {args.output}")


if __name__ == "__main__":
    main()