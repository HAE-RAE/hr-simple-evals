#!/usr/bin/env python
import argparse
import os
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
from dataset_configs import get_config, DATASET_CONFIGS

# LiteLLM을 위한 API 키 설정이 필요하면 여기에서
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

def parse_args():
    """스크립트 실행을 위한 인자들을 파싱합니다."""
    p = argparse.ArgumentParser(description="Run a vLLM model on a dataset and save its responses.")
    p.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    p.add_argument("--dataset", required=True,
                   help=f"Dataset name. Supported: {list(DATASET_CONFIGS.keys())}")
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
    # Use CPU for testing
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", trust_remote_code=True)
    
    # Create a simple wrapper class to mimic vLLM's interface
    class SimpleGenerationOutput:
        def __init__(self, text):
            self.text = text
            
    class SimpleOutput:
        def __init__(self, text):
            self.outputs = [SimpleGenerationOutput(text)]
            
    class SimpleLLM:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.max_length = model.config.max_position_embeddings
            
        def get_tokenizer(self):
            return self.tokenizer
            
        def generate(self, prompts, sampling_params):
            outputs = []
            for prompt in prompts:
                try:
                    # Truncate input if it's too long
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length - sampling_params.max_tokens)
                    
                    output_ids = self.model.generate(
                        inputs.input_ids, 
                        max_new_tokens=sampling_params.max_tokens,
                        do_sample=sampling_params.temperature > 0,
                        temperature=max(sampling_params.temperature, 1e-5),  # Avoid division by zero
                        top_p=sampling_params.top_p
                    )
                    
                    output_text = self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    outputs.append(SimpleOutput(output_text))
                except Exception as e:
                    print(f"Error generating response: {e}")
                    outputs.append(SimpleOutput("Error generating response"))
            return outputs
    
    llm = SimpleLLM(model, tokenizer)

    print(f"📚 데이터셋 로딩 중: {args.dataset_hub_id} - {args.dataset}")
    df = load_dataset(args.dataset_hub_id, args.dataset, split=args.split).to_pandas()
    
    # For testing purposes, use only a small subset of the data
    if len(df) > 5:
        print(f"⚠️ 테스트를 위해 데이터셋을 5개 샘플로 제한합니다 (원래 크기: {len(df)})")
        df = df.head(5)

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