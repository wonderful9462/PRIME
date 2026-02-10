# Following the implementation of https://github.com/open-compass/CompassVerifier/blob/main/src/prompts.py

import os
import re
import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from traceback import format_exc
from concurrent.futures import ThreadPoolExecutor, as_completed

CV_COT_PROMPT = """As a grading expert, your task is to determine whether the candidate's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct
   - The question is perfectly valid - never question them
   - Do not regenerate answers; only compare with the given standard

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure
     * Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components.
     * This determination must be made based on the question's phrasing and the nature of the standard answer.
   - Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors)
   - Disregard any differences in formatting or presentation style
   - For mathematical expressions: calculate step by step whether the two formulas are equivalent
   - For multiple-choice questions: compare only the final choice and corresponding option content

3. Multi-part Answers:
   - For questions requiring multiple responses (e.g., multi-select):
   - All parts must match the standard answer exactly. 
   - Compare each sub-answer step by step. Partial matches are considered incorrect.

4. Validity Check:
   - Reject answers that are:
     * Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) → Label as INCOMPLETE
     * Repetitive (repetition of words or phrases in a loop) → Label as REPETITIVE
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") → Label as REFUSAL
   - For invalid answers, specify the type in the judgment (e.g., \\boxed{{C}} - INCOMPLETE).

Grading Scale:
\\boxed{{A}} - CORRECT: 
   - Answer matches standard exactly (including equivalent expressions)
   - For numerical answers: consider as equivalent if values match when rounded appropriately
   - Semantically equivalent responses

\\boxed{{B}} - INCORRECT:
   - Any deviation from standard answer
   - Partial matches for multi-part questions

\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL)

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate's answer including:
(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition), or a REFUSAL (explicit denial) - if so, immediately classify as \\boxed{{C}} with the corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order, multi-part completeness)
- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs, equivalent but unformatted expressions)
- Required answer type, precision level, etc.
(3) Perform a detailed comparison between the candidate's final answer and the standard answer, for example:
- Content equivalence
- Permitted variations in numerical precision
- Allowed expression formats]
Final Judgment: \\boxed{{A/B/C}} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{reference_answer}
<Standard Answer End>

<Candidate's Answer Begin>
{student_answer}
<Candidate's Answer End>

Analysis step by step and Final Judgment:
"""

def process_judgment(judgment_str: str) -> str:
    # First try to find the exact \boxed{letter} pattern
    boxed_matches = re.findall(r'boxed{([A-C])}', judgment_str)
    if boxed_matches:
        return boxed_matches[-1]
    
    # Directly return the judgment if it is A, B, or C
    if judgment_str in ["A", "B", "C"]:
        return judgment_str
    else:
        final_judgment_str = judgment_str.split("Final Judgment:")[-1]
        matches = re.findall(r'\(([A-C])\)*', final_judgment_str)
        if matches:
            return matches[-1]
        matches = re.findall(r'([A-C])', final_judgment_str)
        if matches:
            return matches[-1]
        return ""

def verify(question, reference, answer, label, **kwargs):
    prompt = CV_COT_PROMPT.format(
        question=question,
        reference_answer=reference,
        student_answer=answer,
    )
    messages = [{"role": "user", "content": prompt}]
    ak, url = kwargs.pop('ak'), kwargs.pop('url')
    for _ in range(3):
        try:
            client = OpenAI(
                api_key=ak,
                base_url=url,
            )
            chat_completion = client.chat.completions.create(
                messages=messages,
                **kwargs,
            )
            token_num = chat_completion.usage.completion_tokens
            response = chat_completion.choices[0].message.content
            judgment = process_judgment(response)
            verdict = judgment == "A"

            return {
                "token_num": token_num,
                "verdict": verdict,
                "correct": verdict == label,
            }
        except Exception:
            print(format_exc())
            time.sleep(1)
    return None

def verify_file(
    src_file,
    output_file,
    concurrent=100,
    **kwargs
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.read_json(src_file, lines=True)
    pbar = tqdm(total=len(df), desc=f"Submitting {os.path.basename(src_file)}")
    executor = ThreadPoolExecutor(max_workers=concurrent)
    futures = {}
    for index, row in df.iterrows():
        question = row['question']
        reference = row['reference']
        answer = str(row['output']).split('</think>')[-1]
        label = row['verdict'] > 0

        future = executor.submit(
            verify,
            question=question,
            reference=reference,
            answer=answer,
            label=label,
            **kwargs,
        )
        futures[future] = index
        pbar.update(1)
    pbar.close()

    all_tokens = []
    all_results = []
    all_details = []
    pbar = tqdm(total=len(df), desc=f"Verifying {os.path.basename(src_file)}")
    try:
        for future in as_completed(futures):
            index = futures[future]
            row = df.iloc[index]
            result = future.result()
            if result is not None:
                tqdm.write(str(result))
                all_tokens.append(result["token_num"])
                all_results.append(result["correct"])
                all_details.append({
                    "idx": row['idx'],
                    "token_num": result["token_num"],
                    "verdict": result["verdict"],
                    "correct": result["correct"],
                    "question": row['question'],
                    "output": row['output'],
                    "reference": row['reference'],
                    "subject": row['subject'],
                })
            futures.pop(future)
            pbar.update(1)
    except KeyboardInterrupt:
        pass
    finally:
        pbar.close()
        print(f"Average tokens: {sum(all_tokens)}/{len(all_tokens)} = {sum(all_tokens)/len(all_tokens)}")
        print(f"Accuracy: {sum(all_results)}/{len(all_results)} = {sum(all_results)/len(all_results)}")

        # Save results to file
        if output_file:
            with open(output_file, 'w') as f:
                for detail in all_details:
                    f.write(json.dumps(detail, ensure_ascii=False) + '\n')
            print(f"Results saved to {output_file}")


if __name__ == "__main__":
    ak = "YOUR_API_KEY"
    model = "CompassVerifier_MODEL"
    url = "REQUEST_URL"

    src_file = f"./data/PRIME.jsonl"
    output_file = f"./results/{model}_results.jsonl"

    verify_file(
        src_file,
        output_file,
        concurrent=128,
        ak=ak,
        url=url,
        model=model,
        timeout=1800,
        temperature=0.0,
    )
