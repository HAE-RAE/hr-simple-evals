import re
import json
import pandas as pd
from functools import partial
from litellm import batch_completion
from tqdm import tqdm

# math_verify 라이브러리를 import하고, 없을 경우를 대비하여 가용성 플래그를 설정합니다.
try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
    print("✅ 'math_verify' 라이브러리를 성공적으로 불러왔습니다.")
except ImportError:
    print("⚠️ 'math_verify' 라이브러리를 찾을 수 없습니다. pip install math-verify 를 권장합니다.")
    MATH_VERIFY_AVAILABLE = False


# ===================================================================
# 평가 유틸리티 함수들
# ===================================================================

def _parse_math_answer(response: str) -> str:
    """수학 문제 응답에서 \\boxed{} 안의 최종 답안을 추출합니다."""
    match = re.search(r"\\boxed\{(.*?)\}", response)
    if match:
        return match.group(1).strip()
    # \\boxed{} 형식이 없을 경우, 응답의 마지막 숫자(정수/실수)를 추출합니다.
    numbers = re.findall(r"[-+]?\d*\.?\d+", response)
    return numbers[-1] if numbers else None

def _normalize_mqa_choice_advanced(response: str) -> str:
    """
    모델의 다양한 객관식 답변 형식(<think> 태그, 숫자 답변 등)을
    정규화하여 최종 선택지를 추출합니다.
    """
    if not isinstance(response, str):
        return None

    # <think> 태그가 있을 경우, 그 이후의 내용만 사용합니다.
    if '</think>' in response:
        response = response.split('</think>', 1)[1]

    # "정답은 [X]입니다" 형식에서 선택지를 우선 추출합니다.
    match = re.search(r"정답은\s*([A-Ja-j1-5])\s*(?:입니다|이다)", response)
    if match:
        return match.group(1).upper().strip(".)")

    # 응답 텍스트에서 마지막으로 언급된 선택지(A-E, 1-5)를 추출합니다.
    choices = re.findall(r'[A-Ea-e1-5]', response)
    if choices:
        last_choice = choices[-1]
        # 숫자 형식의 답변을 알파벳으로 변환합니다 (예: '1' -> 'A').
        if last_choice.isdigit():
            return chr(ord('A') + int(last_choice) - 1)
        return last_choice.upper()

    return response.strip().upper().strip(".)")

def _parse_answer_from_prompt(response: str, task="kobalt") -> str:
    """Kobalt 데이터셋의 특정 답변 형식에서 정답을 추출합니다."""
    if task == "kobalt":
        match = re.search(r"정답은\s*([A-J])\s*입니다", response)
        if match:
            return match.group(1).strip()
    return None

# ===================================================================
# 1. 프롬프트 생성 함수 (Prompt Makers)
# ===================================================================

def _create_prompt_for_mqa(row, tokenizer):
    """객관식 문제(KMMLU, kmmlu-pro)를 위한 프롬프트를 생성합니다."""
    # choices 컬럼이 있는 경우와 없는 경우를 모두 처리
    if 'choices' in row and row['choices']:
        query = (
            f"다음은 '{row['category']}' 분야의 객관식 문제입니다. "
            f"가장 적절한 선택지를 하나만 골라 그 알파벳을 답해주세요.\n\n"
            f"문제: {row['question']}\n\n"
            "선택지:\n"
            f"{row['choices']}\n\n"
            "정답:"
        )
    else:
        # choices 컬럼이 없는 경우 (KoSimpleEval 등), question에 선택지가 포함되어 있음
        query = (
            f"다음은 '{row['category']}' 분야의 객관식 문제입니다. "
            f"가장 적절한 선택지를 하나만 골라 그 번호나 알파벳을 답해주세요.\n\n"
            f"{row['question']}\n\n"
            "정답:"
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True
    )

def _create_prompt_for_kobalt(row, tokenizer):
    """Kobalt 데이터셋을 위한 프롬프트를 생성합니다."""
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
    """수학 문제(AIME, MCLM)를 위한 프롬프트를 생성합니다."""
    question_text = row.get('question', row.get('problem', ''))
    query = (
        f"다음 수학 문제를 풀어주세요. 풀이 과정과 함께 최종 답을 '\\boxed{{정답}}' 형식으로 명확하게 제시해주세요.\n\n"
        f"문제: {question_text}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True
    )

def _create_prompt_for_qa(row, tokenizer):
    """일반적인 질의응답 데이터셋을 위한 프롬프트를 생성합니다."""
    query = f"다음 질문에 대해 상세하고 정확하게 답변해주세요.\n\n질문: {row['question']}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True
    )

def _create_prompt_for_arena(row, tokenizer):
    """Arena 형식 데이터셋을 위한 프롬프트를 생성합니다."""
    query = row['prompts']
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True
    )

# ===================================================================
# 2. 평가 실행 함수 (Evaluators)
# ===================================================================

def _evaluate_mqa(df, args):
    """객관식 문제(KMMLU 등)를 정교한 파싱 로직을 사용하여 평가합니다."""
    print("🤖 객관식 문제(MQA) 평가를 시작합니다 (Advanced Parsing)...")

    df['pred_choice'] = df['response'].apply(_normalize_mqa_choice_advanced)
    df['gold_choice'] = df['gold'].astype(str).str.upper().str.strip(".)")
    df['correct'] = (df['pred_choice'] == df['gold_choice'])

    accuracy = df['correct'].mean()
    print(f"✅ 평가 완료! 전체 정확도: {accuracy:.2%}")
    return df

def _evaluate_math(df, args):
    """수학 문제를 `math_verify` 라이브러리를 사용하여 평가합니다."""
    print("🤖 수학 문제 평가를 시작합니다 (Math-Verify Logic)...")

    if MATH_VERIFY_AVAILABLE:
        correct_output = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Math-Verify 검증"):
            is_correct = False
            try:
                gold_answer = str(row.get('gold', row.get('answer')))
                parsed_gold = parse(gold_answer)
                parsed_pred = parse(row['response'])
                is_correct = verify(parsed_gold, parsed_pred)
            except Exception:
                is_correct = False
            correct_output.append(is_correct)
        df['correct'] = correct_output
    else:
        # math_verify가 없을 경우, 기존의 단순 추출 방식으로 대체합니다.
        print("... 'math_verify'가 없어 기존 방식으로 채점합니다.")
        df['pred_answer'] = df['response'].apply(_parse_math_answer)
        if 'gold' in df.columns:
            df['gold_answer'] = df['gold'].astype(str)
        elif 'answer' in df.columns:
            df['gold_answer'] = df['answer'].astype(str)
        else:
            df['gold_answer'] = 'N/A'
        df['correct'] = (df['pred_answer'] == df['gold_answer'])

    accuracy = df['correct'].mean()
    print(f"✅ 평가 완료! 전체 정확도: {accuracy:.2%}")
    return df

def _evaluate_hrm8k_ksm(df, args):
    """hrm8k-ksm 데이터셋을 평가합니다. 'answer' 컬럼을 'gold'로 변경 후 수학 평가를 실행합니다."""
    print("🤖 hrm8k-ksm 데이터셋 평가를 시작합니다...")
    if 'answer' in df.columns:
        df.rename(columns={'answer': 'gold'}, inplace=True)
    else:
        print("⚠️ 'answer' 컬럼을 찾을 수 없습니다. 평가를 건너뜁니다.")
        return df
    return _evaluate_math(df, args)

def _evaluate_kobalt(df, args):
    """Kobalt 데이터셋을 평가합니다."""
    print("🤖 Kobalt 평가를 시작합니다...")
    df['pred_choice'] = df['response'].apply(partial(_parse_answer_from_prompt, task="kobalt"))
    df['gold_choice'] = df['gold'].astype(str).str.upper().str.strip(".)")
    df['correct'] = (df['pred_choice'] == df['gold_choice'])
    accuracy = df['correct'].mean()
    print(f"✅ 평가 완료! 전체 정확도: {accuracy:.2%}")
    return df

def _evaluate_gpqa_as_math(df, args):
    """gpqa-diamond 데이터셋을 수학 문제처럼 평가합니다."""
    print("🤖 gpqa-diamond 데이터셋을 수학 문제로 평가합니다...")
    if 'Correct Answer' in df.columns:
        df = df.rename(columns={'Correct Answer': 'gold'})
    else:
        print("⚠️ 'Correct Answer' 컬럼을 찾을 수 없습니다. 평가를 건너뜁니다.")
        return df
    return _evaluate_math(df, args)

def _evaluate_arena(df, args):
    """Arena 데이터셋을 외부 LLM을 이용해 평가합니다."""
    print("🤖 Arena 평가를 시작합니다 (Judge 모델: gpt-4-turbo, 방식: JSON Mode)...")
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
        ref_response = row.get('ref', 'N/A')
        query_content = f"### Instruction:\n{row['prompts']}\n\n### Response A:\n{ref_response}\n\n### Response B:\n{row['response']}"
        qrys.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_content}
        ])

    responses = batch_completion(
        model='gpt-4.1-mini',
        messages=qrys,
        response_format={"type": "json_object"}
    )
    score_map = {
        'A>>B': 0, 'A>B': 2, 'A=B': 3, 'A<B': 4, 'A<<B': 5,
        'B>>A': 5, 'B>A': 4, 'B=A': 3, 'B<A': 2, 'B<<A': 1
    }
    verdicts, scores = [], []
    for res in tqdm(responses, desc="심판 응답 파싱 및 채점"):
        try:
            judge_json = json.loads(res.choices[0].message.content)
            verdict = judge_json.get('verdict')
            verdicts.append(verdict)
            scores.append(score_map.get(verdict))
        except (json.JSONDecodeError, AttributeError):
            verdicts.append("PARSE_ERROR")
            scores.append(None)

    df['judge_verdict'] = verdicts
    df['score'] = scores
    avg_score = df['score'].mean(skipna=True)
    error_count = df['score'].isna().sum()
    print(f"✅ Arena 평가 완료! 평균 점수: {avg_score:.3f} (오류: {error_count}개)")
    return df

# ===================================================================
# 3. 데이터셋 설정 종합 (메인 컨트롤)
# ===================================================================
DATASET_CONFIGS = {
    # 기존 설정들
    'KMMLU_Redux': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'ClinicalQA':  {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'KMMLU-Pro': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa}, 
    'KMMLU-HARD': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa}, 
    'KorMedLawQA': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa}, 
    'MCLM': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
    'ArenaHard': {'prompt_maker': _create_prompt_for_arena, 'evaluator': _evaluate_arena},
    'aime_2025': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
    'aime_2024': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
    'default': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
    'train': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
    'click': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'kobalt': {'prompt_maker': _create_prompt_for_kobalt, 'evaluator': _evaluate_kobalt},
    'KSM': {'prompt_maker': _create_prompt_for_qa, 'evaluator': _evaluate_hrm8k_ksm},
    'gpqa-diamond': {'prompt_maker': _create_prompt_for_qa, 'evaluator': _evaluate_gpqa_as_math},
    
    # KoSimpleEval 데이터셋 config들 추가
    'HRB1_0': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'KMMLU': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'CLIcK': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'GPQA': {'prompt_maker': _create_prompt_for_mqa, 'evaluator': _evaluate_mqa},
    'KoBALT-700': {'prompt_maker': _create_prompt_for_kobalt, 'evaluator': _evaluate_kobalt},
    'AIME2024': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
    'AIME2025': {'prompt_maker': _create_prompt_for_math, 'evaluator': _evaluate_math},
}

def get_config(dataset_name):
    """데이터셋 이름에 맞는 프롬프트 생성 함수와 평가 함수 설정을 반환합니다."""
    config = DATASET_CONFIGS.get(dataset_name)
    if config is None:
        raise ValueError(f"'{dataset_name}'에 대한 설정을 찾을 수 없습니다. `dataset_configs.py`에 추가해주세요.")
    return config