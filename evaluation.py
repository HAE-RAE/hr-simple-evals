#!/usr/bin/env python
import argparse
import os
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
from dataset_configs import get_config, DATASET_CONFIGS

# Transformers configuration 중복 등록 문제 해결을 위한 import
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# LiteLLM을 위한 API 키 설정이 필요하면 여기에서
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Transformers configuration 중복 등록 방지를 위한 환경 변수 설정
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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

def _handle_duplicate_config_registration(model_path: str) -> None:
    """중복된 configuration 등록 문제를 해결합니다."""
    try:
        # 모델 경로에서 모델 이름 추출
        model_name = model_path.split('/')[-1].lower()
        
        # 일반적인 aimv2 관련 등록도 확인
        problematic_names = ['aimv2', 'aim_v2', 'aimv2_config', model_name]
        
        # CONFIG_MAPPING에서 중복 등록 제거
        for name in problematic_names:
            try:
                # _extra_content 속성이 있는 경우
                if hasattr(CONFIG_MAPPING, '_extra_content') and name in CONFIG_MAPPING._extra_content:
                    print(f"⚠️ '{name}' configuration이 _extra_content에 이미 등록되어 있습니다. 기존 등록을 제거합니다.")
                    del CONFIG_MAPPING._extra_content[name]
                
                # 직접 CONFIG_MAPPING에 등록된 경우
                if name in CONFIG_MAPPING:
                    print(f"⚠️ '{name}' configuration이 CONFIG_MAPPING에 이미 등록되어 있습니다. 기존 등록을 제거합니다.")
                    # LazyConfigMapping에서는 직접 삭제가 안될 수 있으므로 다른 방법 시도
                    try:
                        del CONFIG_MAPPING[name]
                    except (TypeError, AttributeError):
                        # LazyConfigMapping의 내부 구조에 직접 접근
                        if hasattr(CONFIG_MAPPING, '_mapping') and name in CONFIG_MAPPING._mapping:
                            del CONFIG_MAPPING._mapping[name]
                        if hasattr(CONFIG_MAPPING, '_extra_content') and name in CONFIG_MAPPING._extra_content:
                            del CONFIG_MAPPING._extra_content[name]
            except Exception as e:
                print(f"⚠️ '{name}' configuration 제거 중 오류 (무시하고 계속): {e}")
                
        # AutoConfig의 _model_type_to_module_name에서도 제거
        try:
            from transformers.models.auto.configuration_auto import _model_type_to_module_name
            for name in problematic_names:
                if name in _model_type_to_module_name:
                    print(f"⚠️ '{name}' configuration이 _model_type_to_module_name에 이미 등록되어 있습니다. 기존 등록을 제거합니다.")
                    del _model_type_to_module_name[name]
        except (ImportError, AttributeError):
            pass
            
        # 추가적인 등록 위치들도 확인
        try:
            from transformers.models.auto import modeling_auto
            if hasattr(modeling_auto, 'MODEL_MAPPING'):
                for name in problematic_names:
                    if name in modeling_auto.MODEL_MAPPING:
                        print(f"⚠️ '{name}' configuration이 MODEL_MAPPING에 이미 등록되어 있습니다. 기존 등록을 제거합니다.")
                        try:
                            del modeling_auto.MODEL_MAPPING[name]
                        except Exception:
                            pass
        except (ImportError, AttributeError):
            pass
                
    except Exception as e:
        print(f"⚠️ Configuration 등록 처리 중 오류 발생 (무시하고 계속): {e}")

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
    
    # 2. 중복 configuration 등록 문제 해결
    _handle_duplicate_config_registration(args.model)
    
    print(f"🚀 모델 로딩 중: {args.model}")
    
    # 모델 로딩을 여러 번 시도 (configuration 등록 문제 해결을 위해)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            llm = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
            tokenizer = llm.get_tokenizer()
            break
        except ValueError as e:
            if "already used by a Transformers config" in str(e) and attempt < max_retries - 1:
                print(f"⚠️ 시도 {attempt + 1}/{max_retries}: Configuration 중복 등록 오류 발생. 다시 시도합니다...")
                _handle_duplicate_config_registration(args.model)
                continue
            else:
                print(f"❌ 모델 로딩 실패: {e}")
                return
        except Exception as e:
            print(f"❌ 모델 로딩 중 예상치 못한 오류: {e}")
            return

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