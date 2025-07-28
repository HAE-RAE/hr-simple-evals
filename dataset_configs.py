import re
import json
import pandas as pd
from functools import partial
from litellm import batch_completion
from tqdm import tqdm

# ===================================================================
# 평가 유틸리티 함수들
# ===================================================================

def _parse_math_answer(response: str) -> str:
    """수학 문제 응답에서 \\boxed{} 안의 최종 답안을 추출"""
    match = re.search(r"\\boxed\{(.*?)\}", response)
    if match:
        return match.group(1).strip()
    # \\boxed{} 형식이 없을 경우 -> 걍 실패인가? 일단 응답의 마지막 숫자(정수/실수)를 추출.
    numbers = re.findall(r"[-+]?\d*\.?\d+", response) # 이렇게 해도 잘 안나올 것 같긴 함. 필요없으면 제거
    return numbers[-1] if numbers else None

def _normalize_mqa_choice(choice: str) -> str:
    """객관식 문제의 선택지를 정규화(예: 'A)', 'B.' -> 'A', 'B')"""
    if not isinstance(choice, str):
        return None
    return choice.strip().upper().strip(".)")

def _parse_answer_from_prompt(response: str, task="kobalt") -> str:
    if task == "kobalt":
        match = re.search(r"정답은\s*([A-J])\s*입니다", response)
        if match:
            return match.group(1).strip()
    return None  # TODO: add parsing for unknown

# ===================================================================
# 1. 프롬프트 생성 함수 (Prompt Makers)
# ===================================================================
# 각 함수는 데이터프레임의 행(row)과 토크나이저(tokenizer)를 받아 모델에 입력될 최종 프롬프트 문자열을 반환하는 역할

def _create_prompt_for_mqa(row, tokenizer):
    """객관식 문제(KMMLU, kmmlu-pro)를 위한 프롬프트"""
    # TODO: 데이터셋의 'choices' 컬럼 포맷에 맞춰 프롬프트 세부 조정 필요 -> KMMLU, KMMLU-Pro만이면 크게 상관 없을지도
    # 예: A. {row.choices[0]}\nB. {row.choices[1]}...
    query = (
        f"다음은 '{row['category']}' 분야의 객관식 문제입니다. "
        f"가장 적절한 선택지를 하나만 골라 그 알파벳을 답해주세요.\n\n"
        f"문제: {row['question']}\n\n"
        "선택지:\n"
        f"{row['choices']}\n\n" # 'choices' 컬럼이 선택지를 포함한 문자열이라고 가정
        "정답:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True
    )

def _create_prompt_for_kobalt(row, tokenizer):
    """Kobalt (https://arxiv.org/pdf/2505.16125) Figure 2."""
    query = (
        f"다음 문제에 대해서 충분히 생각하고 추론하여, 10개의 보기(A, B, C, D, E, F, G, H, I, J) 중 정답을 고르세요.\n"
        f"{row['question']}\n\n"
        f"답변은 반드시 다음 형식을 엄격히 지켜야 합니다: \"정답은 [정답 보기]입니다.\"로 끝나야하고, [정답 보기]는 A, B, C, D, E, F, G, H, I, J 중 하나여야 합니다.\n"
        f"문제를 풀기 위해, 한번 천천히 생각해봅시다."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True
    )

def _create_prompt_for_math(row, tokenizer):
    """수학 문제(AIME, MCLM)를 위한 프롬프트"""
    query = (
        f"다음 수학 문제를 풀어주세요. 풀이 과정과 함께 최종 답을 '\\boxed{{정답}}' 형식으로 명확하게 제시해주세요.\n\n"
        f"문제: {row['question']}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True
    )

def _create_prompt_for_qa(row, tokenizer):
    """일반적인 질의응답 데이터셋을 위한 프롬프트 (GPQA, KoBALT 등)"""
    query = f"다음 질문에 대해 상세하고 정확하게 답변해주세요.\n\n질문: {row['question']}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True
    )

def _create_prompt_for_arena(row, tokenizer):
    """Arena 형식 데이터셋을 위한 프롬프트"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": row['prompts']}],
        tokenize=False, add_generation_prompt=True
    )

# ===================================================================
# 2. 평가 실행 함수 (Evaluators)
# ===================================================================
# 각 함수는 모델 응답이 추가된 데이터프레임(df)과 실행 인자(args)를 받아 채점 결과가 추가된 데이터프레임을 반환하는 역할.

def _evaluate_mqa(df, args):
    """객관식 문제(KMMLU, kmmlu-pro) 평가"""
    print("🤖 객관식 문제(MQA) 평가를 시작합니다...")

    df['pred_choice'] = df['response'].apply(_normalize_mqa_choice)
    df['gold_choice'] = df['gold'].apply(_normalize_mqa_choice)
    df['correct'] = (df['pred_choice'] == df['gold_choice'])

    accuracy = df['correct'].mean()
    print(f"✅ 평가 완료! 전체 정확도: {accuracy:.2%}")
    return df

def _evaluate_math(df, args):
    """수학 문제(AIME, MCLM) 평가"""
    print("🤖 수학 문제 평가를 시작합니다...")

    df['pred_answer'] = df['response'].apply(_parse_math_answer)
    df['gold_answer'] = df['gold'].astype(str) # 정답을 문자열로 통일

    df['correct'] = (df['pred_answer'] == df['gold_answer'])

    accuracy = df['correct'].mean()
    print(f"✅ 평가 완료! 전체 정확도: {accuracy:.2%}")
    return df

def _evaluate_hrm8k_ksm(df, args):
    """hrm8k-ksm 평가를 위해 'answer' 컬럼을 'gold'로 변경 후 math 평가 실행"""
    print("🤖 hrm8k-ksm 데이터셋 평가를 시작합니다 (컬럼명 변경 후)...")
    # 'answer' 컬럼을 'gold'로 이름을 변경합니다.
    if 'answer' in df.columns:
        df.rename(columns={'answer': 'gold'}, inplace=True)
    else:
        print("⚠️ 'answer' 컬럼을 찾을 수 없습니다. 평가를 건너뜁니다.")
        return df

    # 기존 수학 평가 함수를 호출합니다.
    return _evaluate_math(df, args)

def _evaluate_kobalt(df, args):
    """Kobalt 평가"""
    print("🤖 Kobalt 평가를 시작합니다...")
    # TODO: debug
    df['pred_choice'] = df['response'].apply(partial(_parse_answer_from_prompt, task="kobalt"))
    df['gold_choice'] = df['gold'].apply(_normalize_mqa_choice)
    df['correct'] = (df['pred_choice'] == df['gold_choice'])
    accuracy = df['correct'].mean()
    print(f"✅ 평가 완료! 전체 정확도: {accuracy:.2%}")
    return df

# dataset_configs.py 파일의 함수 영역에 추가

def _evaluate_gpqa_as_math(df, args):
    """gpqa-d 평가를 위해 'Correct Answer'를 'gold'로 변경 후 math 평가 실행"""
    print("🤖 gpqa-diamond 데이터셋을 수학 문제로 평가합니다...")

    # 'Correct Answer' 컬럼의 이름을 'gold'로 변경
    if 'Correct Answer' in df.columns:
        # df.rename은 새로운 데이터프레임을 반환하므로 다시 할당해야 함
        df = df.rename(columns={'Correct Answer': 'gold'})
    else:
        print("⚠️ 'Correct Answer' 컬럼을 찾을 수 없습니다. 평가를 건너뜁니다.")
        return df

    # 기존 수학 평가 함수(_evaluate_math)를 호출
    return _evaluate_math(df, args)

def _evaluate_arena(df, args):
    """Arena 데이터셋 평가 (안정적인 JSON 모드 사용)"""
    print("🤖 Arena 평가를 시작합니다 (Judge 모델: gpt-4-turbo, 방식: JSON Mode)...")

    # JSON 출력 형식을 명시적으로 지시하는 시스템 프롬프트
    system_prompt = (
        "You are a helpful and precise assistant for checking the quality of AI responses. "
        "Compare Response A and Response B based on the user's instruction. "
        "Your decision must be one of the following categories: "
        "B>>A, B>A, B=A, A>B, A>>B. "
        "Please provide your response in a JSON object with two keys: "
        "'verdict' (the category of your decision) and 'reasoning' (a brief explanation)."
    )

    qrys = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Arena 평가 질문 생성"):
        # ArenaHard 외 다른 데이터셋 필요 하면 참조 응답(ref) 컬럼명 확인 필요
        ref_response = row.get('ref', 'N/A')
        query_content = f"### Instruction:\n{row['prompts']}\n\n### Response A:\n{ref_response}\n\n### Response B:\n{row['response']}"
        qrys.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_content}
        ])

    responses = batch_completion(
        model='gpt-4-turbo', # 모델 명 바꿀거면 바꾸기
        messages=qrys,
        response_format={"type": "json_object"}
    )

    # 기존 코드의 점수표를 그대로 사용
    score_map = {
        'A>>B': 0, 'A>B': 2, 'A=B': 3, 'A<B': 4, 'A<<B': 5, # A<B, A<<B는 B>A, B>>A와 동일
        'B>>A': 5, 'B>A': 4, 'B=A': 3, 'B<A': 2, 'B<<A': 1
    }

    verdicts = []
    scores = []
    for res in tqdm(responses, desc="심판 응답 파싱 및 채점"):
        try:
            judge_json = json.loads(res.choices[0].message.content)
            verdict = judge_json.get('verdict')
            verdicts.append(verdict)
            scores.append(score_map.get(verdict))
        except (json.JSONDecodeError, AttributeError):
            # 파싱 실패
            verdicts.append("PARSE_ERROR")
            scores.append(None)

    df['judge_verdict'] = verdicts
    df['score'] = scores

    avg_score = df['score'].mean(skipna=True)
    error_count = df['score'].isna().sum()

    print(f"✅ Arena 평가 완료! 평균 점수: {avg_score:.3f} (오류: {error_count}개)")
    return df

def _evaluate_placeholder(df, args):
    """구현이 필요한 데이터셋을 위한 임시 평가 함수""" # 아직 안된애들에 꽂아두기 용
    print(f"⚠️  '{args.dataset}' 데이터셋의 평가 로직이 구현되지 않았습니다. 결과 파일만 생성합니다.")
    # TODO: 해당 데이터셋에 맞는 평가 로직을 구현 필요
    df['correct'] = 'N/A'
    return df

# ===================================================================
# 3. 데이터셋 설정 종합 (메인 컨트롤)
# ===================================================================
DATASET_CONFIGS = {
    'KMMLU-Redux': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'MCLM': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
    'ArenaHard': {'prompt_maker': _create_prompt_for_arena, 'evaluator': _evaluate_arena},
    'kmmlu-pro': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'aime2025': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
    'aime2024': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
    'click': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'kobalt': {'prompt_maker': _create_prompt_for_kobalt, 'evaluator': _evaluate_kobalt},
    'KSM': {'prompt_maker': _create_prompt_for_qa, 'evaluator': _evaluate_hrm8k_ksm},
    'gpqa-diamond': {'prompt_maker': _create_prompt_for_qa, 'evaluator': _evaluate_gpqa_as_math},
}

def get_config(dataset_name):
    """데이터셋 이름에 맞는 설정(프롬프트 생성 함수, 평가 함수)을 반환데스."""
    config = DATASET_CONFIGS.get(dataset_name)
    if config is None:
        raise ValueError(f"'{dataset_name}'에 대한 설정을 찾을 수 없습니다. `dataset_configs.py`에 추가해주세요.")
    return config
