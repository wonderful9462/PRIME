# Following the implementation of https://github.com/IAAR-Shanghai/xVerify/blob/main/src/xVerify/prompts/judge_prompt.py
import os
import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from traceback import format_exc
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{student_answer}"""

Correct answer: {reference_answer}

Judgement:
'''

def verify(question, reference, answer, label, **kwargs):
    prompt = PROMPT.format(
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
            if response.strip().lower() == "incorrect":
                verdict = False
            else:
                verdict = True

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
    model = "xverify-9b"
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
    )
