import pandas as pd
import pickle

from sklearn.metrics import accuracy_score


df = pd.read_json('liar_dataset/LIAR_test.jsonl', lines=True, orient='records')

save_list_list = []
for save_part in [0,1,2,3,4,5,6,7,8,9,10,11,12,13]:
    with open(f'LIAR_test_gpt4_part{save_part}.pkl', 'rb') as f:
        save_list_list.append(pickle.load(f))

#combined = save_list_list[0] + save_list_list[1] + save_list_list[2][1:] + save_list_list[4] + save_list_list[3]
combined = []
for i in save_list_list:
    combined += i

print(len(combined), 'total examples')

response_list = []
completion_tokens = 0
prompt_tokens = 0
for i in combined:
    response_list.append(i["choices"][0]["message"]["content"])
    completion_tokens += i["usage"]["completion_tokens"]
    prompt_tokens += i["usage"]["prompt_tokens"]

print(completion_tokens, prompt_tokens)

df['gpt4-answer'] = response_list

df.to_json('LIAR_test_gpt4preds.jsonl', lines=True, orient='records')





