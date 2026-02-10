import os
import re
import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from traceback import format_exc
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = """# Role

You are an extremely meticulous and conscientious exam grader. Your task is to assess whether a student’s submitted answer is correct.

# Input Variables

- **[Question]**: The problem that needs to be solved.

- **[Reference Answer]**: Typically only the final result, as a reference.

- **[Student Answer]**: Includes the full solution steps and the final result.

# Primary Principles

I will provide you with a question and the corresponding reference answer. You need to determine:

1. Whether the student’s solution **process** is correct.

2. Whether the student’s final **result** is correct.

3. Whether the student’s answer is **perfect**.

# Core Logic
## Process Correctness

You must carefully check whether there are problems in each step of the student’s solution.
Your judgment should follow the principle of “presumption of correctness”: the process is considered correct unless one of the following fundamental errors occurs.

**Fundamental Errors**:

1. Logical errors, calculation errors, or factual errors in reasoning.

2. Inconsistencies within reasoning, such as inconsistency between statements.

**Note**: If the student did not provide the solution process, it is considered **correct by default**.

## Result Correctness

Similarly, your judgment of the final result also follows the principle of “presumption of correctness”: the answer is considered correct unless one of the following fundamental errors occurs.

**Fundamental Errors**:

1. The student fails to provide a clear final answer.

2. The student’s final answer does not satisfy the problem requirements.

3. The student’s final result does not match the reference answer (unless you can prove the student’s result is also correct).

**Note**: In numerical calculations, discrepancies due to reasonable rounding are **not considered** fundamental errors but should be carefully distinguished from computational errors.

## Answer Perfection

An answer is considered perfect only when both the process and result are correct, and none of the following imperfections occur. Any answer with a fundamental error is necessarily imperfect.

Imperfection reference standards:

1. Insufficient precision: The solution does not provide the exact form (e.g., √2) but only an approximate value (e.g., 1.414).

2. Not simplified: Fractions, radicals, etc., are not expressed in simplest form.

3. Redundant content: Contains information clearly irrelevant to the solution.

4. Lack of process: Missing necessary solution steps.

5. Other flaws you consider imperfect but not fundamental errors.

**Note**: If the problem’s requirements conflict with the standards of perfection, the problem requirements take precedence.

# Special Cases

- If the student’s answer contains a large amount of repetitive content, both the process and the result should be considered incorrect.

- If the student’s answer is incomplete (e.g., cut off), both the process and the result should be considered incorrect.

# Output Format

You need to output a result in XML format with the following fields:

<process> [Your judgment on the correctness of the solution process, only True or False] </process>
<result> [Your judgment on the correctness of the final result, only True or False] </result>
<perfect> [Your judgment on whether the answer is perfect, only True or False] </perfect>
<reason> [Your reasoning for the judgment in a few brief sentences] </reason>
{few_shots}
# Task

Now, please strictly follow the above principles, logic, and format to evaluate the following input and return the result in XML format.

<question>
{question}
</question>

<student_answer>
{student_answer}
</student_answer>

<reference_answer>
{reference_answer}
</reference_answer>
"""

FEW_SHOTS = """
# Case Analysis

**Case 1**:

- **[Question]**: There are 9 apples in the store. 3 were sold, and then 5 were restocked. How many apples are there now?

- **[Reference Answer]**: 11

- **[Student Answer]**: First, the store sold 3 apples, leaving 9-3=6 apples. Then, 5 apples were restocked, so there are now 6+5=10 apples. The final answer is 10.

- **Your Analysis**:

1. **Process Check**: The student made a calculation error. 6+5 should be 11, not 10, so the process is incorrect.

2. **Result Check**: The student’s answer of 10 is inconsistent with the reference answer of 11, so the final result is incorrect.

3. **Is It Perfect**: Both the process and result have fundamental errors, so the answer is not perfect.

**Case 2**:

- **[Question]**: There are 9 apples in the store. 3 were sold, and then 5 were restocked. How many apples are there now?

- **[Reference Answer]**: 11

- **[Student Answer]**: Addition of integers forms an Abel group. To calculate how many apples there are, we need to compute 9-3+5=11, so there are 11 apples in total.

- **Your Analysis**:

1. **Process Check**: The student included some unrelated content, but there were no factual or computational errors, so the process is correct.

2. **Result Check**: The student’s final answer matches the reference answer, so the result is correct.

3. **Is It Perfect**: The process included irrelevant content, so the answer is not perfect.

**Case 3**:

- **[Question]**: There are 9 apples in the store. 3 were sold, and then 5 were restocked. How many apples are there now?

- **[Reference Answer]**: 11

- **[Student Answer]**: Let me try, 9-3+5=11, so the answer might be 11? I guess that’s it.

- **Your Analysis**:

1. **Process Check**: There are no fundamental errors in the solution process, so the process is correct.

2. **Result Check**: The student’s statement of uncertainty constitutes “unable to give a clear conclusion,” which is considered an error, so the result is incorrect.

3. **Is It Perfect**: The result has a fundamental error, so the answer is not perfect.

**Case 4**:

- **[Question]**: In an isosceles right triangle, both legs have a length of 1. What is the length of the hypotenuse?

- **[Reference Answer]**: √2

- **[Student Answer]**: To calculate the length of the hypotenuse, we can use the Pythagorean theorem: hypotenuse c = √(1² + 1²) = 1.414.

- **Your Analysis**:

1. **Process Check**: The student correctly applied the Pythagorean theorem, and the approximation is consistent with rounding rules, so the process is correct.

2. **Result Check**: The student’s result, when rounded, matches the reference answer, so the result is correct.

3. **Is It Perfect**: The student did not provide the exact radical form, so the answer is not perfect.

**Case 5**:

- **[Question]**: In an isosceles right triangle, both legs have a length of 1. What is the length of the hypotenuse?

- **[Reference Answer]**: √2

- **[Student Answer]**: To calculate the length of the hypotenuse, we can use the Pythagorean theorem: hypotenuse c = √(1² + 1²) = √2, so the hypotenuse is approximately 1.414.

- **Your Analysis**:

1. **Process Check**: The student correctly applied the Pythagorean theorem, and the approximation is consistent with rounding rules, so the process is correct.

2. **Result Check**: The student’s result, when rounded, matches the reference answer, so the result is correct.

3. **Is It Perfect**: The student provided the exact radical form, so the answer is perfect.

**Case 6**:

- **[Question]**: In an isosceles right triangle, both legs have a length of 1. What is the length of the hypotenuse? (Round to three decimal places)

- **[Reference Answer]**: 1.414

- **[Student Answer]**: To calculate the length of the hypotenuse, we can use the Pythagorean theorem: hypotenuse c = √(1² + 1²) = √2.

- **Your Analysis**:

1. **Process Check**: The student correctly applied the Pythagorean theorem, so the process is correct.

2. **Result Check**: The student’s result is exact but does not meet the requirement to round to three decimal places, which is a fundamental error, so the result is incorrect.

3. **Is It Perfect**: The result is incorrect, so the answer is not perfect.

**Case 7**:

- **[Question]**: What is the capital of France?

- **[Reference Answer]**: Paris

- **[Student Answer]**: The capital of France is Paris.

- **Your Analysis**:

1. **Process Check**: The student did not provide a process, but according to the rules, the process is considered correct.

2. **Result Check**: The student’s result matches the reference answer, so the result is correct.

3. **Is It Perfect**: No reasoning was needed, so the answer is perfect.

**Case 8**:

- **[Question]**: What is the general formula for the Fibonacci sequence?

- **[Reference Answer]**: F(n) = (φ^n - (1-φ)^n) / √5, where φ = (1 + √5) / 2

- **[Student Answer]**: F(n) = (φ^n - (1-φ)^n) / √5, where φ = (1 + √5) / 2

- **Your Analysis**:

1. **Process Check**: The student did not provide a reasoning process, but according to the rules, the process is considered correct.

2. **Result Check**: The student’s result matches the reference answer, so the result is correct.

3. **Is It Perfect**: Normally, reasoning is required, so the answer is not perfect.

**Case 9**:

- **[Question]**: Given the chemical reaction A+B→C+D with equilibrium constant k=0.5, what is the equilibrium constant for 2A+2B→2C+2D?

- **[Reference Answer]**: 0.25

- **[Student Answer]**: When the coefficients of a reaction are multiplied by n, the equilibrium constant is raised to the nth power. So the new equilibrium constant is k^2.

- **Your Analysis**:

1. **Process Check**: The student’s solution process is correct.

2. **Result Check**: The student’s result matches the reference answer, so the result is correct.

3. **Is It Perfect**: The final result of 0.25 should have been explicitly stated, so the answer is not perfect.

**Case 10**:

- **[Question]**: There are 9 apples in the store. 3 were sold, and then 5 were restocked. How many apples are there now?

- **[Reference Answer]**: 11

- **[Student Answer]**: There are 12 apples. Let me prove it to you: 9-3+5=11, oh no, actually there are 11 apples.

- **Your Analysis**:

1. **Process Check**: The student’s answer contains an inconsistency in reasoning, so the process is incorrect.

2. **Result Check**: The student’s final result matches the reference answer, so the result is correct.

3. **Is It Perfect**: There is a fundamental error in the process, so the answer is not perfect.

**Case 11**:

- **[Question]**: There are 9 apples in the store. 3 were sold, and then 5 were restocked. How many apples are there now?

- **[Reference Answer]**: 11

- **[Student Answer]**: To calculate how many apples are left, we should compute 9-3+5, which equals 11, which equals 11, which equals 11... (repeated many times)

- **Your Analysis**:

1. **Process Check**: The student’s answer includes excessive repetition, so the process is incorrect.

2. **Result Check**: The answer is repetitive, so the result is also incorrect.

3. **Is It Perfect**: Both the process and result are incorrect, so the answer is not perfect.

**Case 12**:

- **[Question]**: There are 9 apples in the store. 3 were sold, and then 5 were restocked. How many apples are there now?

- **[Reference Answer]**: 11

- **[Student Answer]**: To calculate how many apples are left, we should compute 9-3+5, which equals 11. So the store

- **Your Analysis**:

1. **Process Check**: The student’s answer is incomplete, so the process is incorrect.

2. **Result Check**: The answer is incomplete, so the result is also incorrect.

3. **Is It Perfect**: Both the process and result are incorrect, so the answer is not perfect.
"""

def verify(question, reference, answer, label, **kwargs):
    prompt = PROMPT.format(
        question=question,
        reference_answer=reference,
        student_answer=answer,
        few_shots=FEW_SHOTS,
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
            match = re.findall(r".*<process>(.*?)</process>.*<result>(.*?)</result>.*<perfect>(.*?)</perfect>.*<reason>(.*?)</reason>.*", response, re.DOTALL)
            if match:
                process, result, perfect, reason = match[-1]
                process, result, perfect = process.strip().lower(), result.strip().lower(), perfect.strip().lower()
                reason = reason.strip()
                if process not in ["true", "false"] or result not in ["true", "false"] or perfect not in ["true", "false"]:
                    continue
                process = True if process == "true" else False
                result = True if result == "true" else False
                perfect = True if perfect == "true" else False
                verdict = process and result
                return {
                    "token_num": token_num,
                    "correct": verdict == label,
                    "process": process,
                    "result": result,
                    "perfect": perfect,
                    "reason": reason,
                }
        except Exception:
            print(format_exc())
            time.sleep(1)
    return None

def verify_file(
    src_file,
    output_file,
    concurrency=100,
    **kwargs
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.read_json(src_file, lines=True)
    pbar = tqdm(total=len(df), desc=f"Submitting {os.path.basename(src_file)}")
    executor = ThreadPoolExecutor(max_workers=concurrency)
    futures = {}
    for index, row in df.iterrows():
        question = row['question']

        # reference = row['answer']
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
                    "process": result["process"],
                    "result": result["result"],
                    "perfect": result["perfect"],
                    "reason": result["reason"],
                    "token_num": result["token_num"],
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
    model = "EVALUATED_MODEL"
    url = "REQUEST_URL"
    
    src_file = f"./data/PRIME.jsonl"
    output_file = f"./results/{model}_results.jsonl"

    verify_file(
        src_file,
        output_file,
        concurrency=128,
        ak=ak,
        url=url,
        model=model,
        timeout=1800,
        reasoning_effort='medium', # if evaluate gpt-oss series models, set to 'medium'
        temperature=0.6,
    )
