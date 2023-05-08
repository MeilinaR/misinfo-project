import pandas as pd
import tiktoken
import openai
import pickle
import time

OPENAI_API_KEY = 'API_KEY'

openai.api_key = OPENAI_API_KEY

token_input_counter = 0
token_output_counter = 0

encoding = tiktoken.encoding_for_model("gpt-4")


def get_score(text):
    global token_input_counter, token_output_counter

    tmp_prompt = f'Rate the truthfulness of the following statement: "{text}" \
                    Provide a score from 0 to 100, where 0 represents definitively false and 100 represents definitively true. Do not provide any explanations, only respond with the numerical score.'
    token_input_counter += len(encoding.encode(tmp_prompt))

    messages=[{"role": "user", "content": tmp_prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-4", # "gpt-3.5-turbo"
        max_tokens=1000,
        temperature=0.5,
        messages = messages)

    token_output_counter += response["usage"]["completion_tokens"]

    return response

# note: done in multiple parts due to GPT-4 API crashes
df = pd.read_json('liar_dataset/LIAR_train.jsonl', lines=True, orient='records')
statement_list = df.text.tolist()

#save_part = 0 # VAL part1: [:5] part2: [5: ...5+66?] part3: 70:100 part4: 156: part5: 100-156
#save_part = 0 # TEST part0: [:61] part1: [61: 61+52=113] part2: 113:237 part3: 237:253 part4: 253-262 part5: -372 part6: -453 part7: 545 part8: 584 part9: 701 part10 778 p11 856 p12 1178 p13
save_part = 0

response_list = []
total_statements_counter = 0
total_statements_this_block = 0
for statement in statement_list[total_statements_counter:]:     
    print(total_statements_counter, statement)
    gpt_response = get_score(statement)
    response_list.append(gpt_response)
    total_statements_counter += 1
    print(gpt_response["choices"][0]["message"]["content"])
    print(token_input_counter, token_output_counter, total_statements_counter)

    with open(f'LIAR_train_gpt4_part{save_part}_tokenlog.txt', 'a') as f:
        f.write(str(token_input_counter + token_output_counter))

    with open(f'LIAR_train_gpt4_part{save_part}.pkl', 'wb') as f:
        pickle.dump(response_list, f)

    total_statements_this_block += 1
    if total_statements_this_block % 25 == 0:
        time.sleep(70)

print(token_input_counter, token_output_counter)

with open(f'LIAR_train_gpt4_part{save_part}.pkl', 'wb') as f:
    pickle.dump(response_list, f)

