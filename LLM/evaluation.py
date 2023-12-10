# TODO 1: MC -> answer score (EM)
# TODO 2: MC -> explanation score (5-factors)

# TODO 1: SA -> answer score (STS)
# TODO 2: SA -> explanation score (5-factors)

multiple_choice_evaluation_prompt_template = """
You are machine learning professor. 
RESPONSE is consist of LOGICAL REASONING and RESULT.

RESPONSE can splitted into two segments by "Answer:", the first segment become LOGICAL REASONING and second segment is RESULT.


Based on QUESTION, You should decide the RESULT is correct or not, using GOLD ANSWER.
GOLD_ANSWER is always correct

And if the RESULT is correct, you should measure the following metric for the LOGICAL REASONING.

1. Clarity: How clearly is the explanation presented? Can graduate students easily understand the content?
2. Relevance: Is the explanation relevant to the quiz questions and aligned with the expectations for graduate-level knowledge?
3. Clarification of Concepts: Does the explanation clarify required concepts, theories, or knowledge areas relevant to the quiz?
4. Conciseness: Is the explanation concise and focused, avoiding unnecessary details while maintaining completeness?
5. Professional Tone: Is the tone and style of the explanation appropriate for a graduate-level audience, reflecting a high level of professionalism?

QUESTION: {question}

RESPONSE: {pred_response}

GOLD ANSWER: {gold_answer}

if RESULT is same with GOLD ANSWER, print 1, otherwise print 0 into IS_ANSWER_CORRECT
IS_ANSWER_CORRECT: 

In addtion, print scores of above five metrics for LOGICAL REASONING when the RESULT is correct. Each score for five metrics is bounded from 0.00 to 1.00 in two decimal places.
Remember if the RESULT is not correct, you should print None for scores of above five metrics.
No explanation JUST print scores.

Clarity score: 
Relevance score:
Clarification of Concepts score:
Conciseness score:
Professional Tone score:
"""


short_answer_evaluation_prompt_template = """
You are machine learning professor. 
RESPONSE is consist of LOGICAL REASONING and RESULT.

RESPONSE can splitted into two segments by "Answer:", the first segment become LOGICAL REASONING and second segment is RESULT.

Based on QUESTION, You should calculate the sementic similarity score between RESULT and GOLD ANSWER. The score is bounded from 0.00 to 1.00 in two decimal places.

And if sementic similarity score is over than 0.5, you should measure the following metric for the LOGICAL REASONING.

1. Clarity: How clearly is the explanation presented? Can graduate students easily understand the content?
2. Relevance: Is the explanation relevant to the quiz questions and aligned with the expectations for graduate-level knowledge?
3. Clarification of Concepts: Does the explanation clarify required concepts, theories, or knowledge areas relevant to the quiz?
4. Conciseness: Is the explanation concise and focused, avoiding unnecessary details while maintaining completeness?
5. Professional Tone: Is the tone and style of the explanation appropriate for a graduate-level audience, reflecting a high level of professionalism?

QUESTION: {question}
RESPONSE: {pred_response}

GOLD ANSWER: {gold_answer}

In addtion, print scores of above five metrics for LOGICAL REASONING when sementic similarity score is over than 0.5.
Each score for five metrics is bounded from 0.00 to 1.00 in two decimal places.
Remember if sementic similarity score is less than 0.5, you should print None for scores of above five metrics.
No explanation JUST print scores.

Clarity score:
Relevance score:
Clarification of Concepts score:
Conciseness score:
Professional Tone score:
"""

from openai import OpenAI
import openai
import json
import csv
from tqdm import tqdm
client = OpenAI(
  api_key='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'
)

def generate_solutions(model_name, prompt, system_prompt):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},   # professor persona, 
                {"role": "user", "content": prompt}              # lecturenote, qtype, question, strategy
            ]
        ).choices[0].message.content
    except openai.RateLimitError as e:
        from time import sleep
        print(e)
        print('Rate limit exceeded. Sleeping for 10 minute...')
        sleep(600)
        response = generate_solutions(model_name, prompt, system_prompt)

    return response

def get_prompt(mc_template, sa_template, response):
    final_prompt = ''
    is_mc = True if response["question_type"] == "multiple choice" else False
    if is_mc:
        final_prompt = mc_template.format(question=response["query"], pred_response=response["result"], gold_answer=response["answer"])
    else:
        final_prompt =sa_template.format(question=response["query"], pred_response=response["result"], gold_answer=response["answer"])
    
    return final_prompt

system_prompt = "You are machine learning professor."

with open("/home/jinmyeong/code/GPA/LLM/result/few_shot_sft_mistral.json", 'r') as sft:
    sft_responses = json.load(sft)
with open("/home/jinmyeong/code/GPA/LLM/result/few_shot_vanilla_zephyr.json", 'r') as zephyr:
    zephyr_responses = json.load(zephyr)

def get_eval(responses):
    result_list = []
    for response in tqdm(responses):
        response_dict = {}
        prompt = get_prompt(mc_template=multiple_choice_evaluation_prompt_template, sa_template=short_answer_evaluation_prompt_template, response=response)
        result = generate_solutions(model_name="gpt-4-1106-preview", prompt=prompt, system_prompt=system_prompt)
        
        try:
            process_result = [e for e in result.split('\n') if len(e) > 1]
            for e in process_result:
                [key, value] = e.split(": ")
                response_dict[key] = value
        except:
            response_dict = {}

        result_list.append(response_dict)

    return result_list

sft_result = get_eval(sft_responses)
with open("/home/jinmyeong/code/GPA/LLM/result/sft_eval.json", "w") as save_sft:
    json.dump(sft_result, save_sft)

zephyr_result = get_eval(zephyr_responses)
with open("/home/jinmyeong/code/GPA/LLM/result/zephyr_eval.json", "w") as save_zephyr:
    json.dump(zephyr_result, save_zephyr)



with open("/home/jinmyeong/code/GPA/LLM/result/sft_eval.json", "r") as sft:
    sft_eval = json.load(sft)

with open("/home/jinmyeong/code/GPA/LLM/result/zephyr_eval.json", "r") as zephyr:
    zephyr_eval = json.load(zephyr)

def divide_question_type(eval, question_type):
    short_answer_key = 'Semantic similarity score'
    multi_choice_key = 'IS_ANSWER_CORRECT'

    short_answer_list = []
    multi_choice_list = []

    for eval_dict in eval:
        if short_answer_key in eval_dict:
            short_answer_list.append(eval_dict)
        elif multi_choice_key in eval_dict:
            multi_choice_list.append(eval_dict)

    with open(f"/home/jinmyeong/code/GPA/LLM/result/{question_type}_mc_eval.json", "w") as mc:
        json.dump(multi_choice_list, mc)

    with open(f"/home/jinmyeong/code/GPA/LLM/result/{question_type}_sa_eval.json", "w") as sa:
        json.dump(short_answer_list, sa)   

divide_question_type(eval=sft_eval, question_type='sft')
divide_question_type(eval=zephyr_eval, question_type='zephyr')

def get_average(eval_path, type):
    with open(eval_path, "r") as f:
        eval_list = json.load(f)
    keys = list(eval_list[0].keys())
    average_dict = {}
    for key in keys:
        value_list = []
        for eval_dict in eval_list:
            try:
                value_list.append(float(eval_dict[key]))
            except:
                continue
        average_dict[key] = sum(value_list)/len(value_list)

    with open(f"/home/jinmyeong/code/GPA/LLM/result/{type}_average.json", "w") as sa:
        json.dump(average_dict, sa) 
    
    print(average_dict)

get_average(eval_path="/home/jinmyeong/code/GPA/LLM/result/sft_mc_eval.json", type="sft_mc_eval")
get_average(eval_path="/home/jinmyeong/code/GPA/LLM/result/sft_sa_eval.json", type="sft_sa_eval")
get_average(eval_path="/home/jinmyeong/code/GPA/LLM/result/zephyr_mc_eval.json", type="zephyr_mc_eval")
get_average(eval_path="/home/jinmyeong/code/GPA/LLM/result/zephyr_sa_eval.json", type="zephyr_sa_eval")

