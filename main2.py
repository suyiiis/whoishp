# 使用pipeline加载模型
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import json
from datasets import load_dataset, Dataset
import tqdm

model_path = "microsoft/Llama2-7b-WhoIsHarryPotter"
tokenizer_path = "microsoft/Llama2-7b-WhoIsHarryPotter"
torch.cuda.empty_cache()
generator = pipeline("text-generation",
                     model=model_path,
                     tokenizer=model_path,
                     device=6)
print("model loaded")
tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)
generator.tokenizer.pad_token_id = generator.model.config.eos_token_id


def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list


bad_words_ids = get_tokens_as_list(word_list=["\n"])

# 此处不使用chat模型进行文本生成
# file_path

file_path = 'hp_score_true.txt'
# read file to var hp_text
with open(file_path, 'r', encoding='utf-8') as f:
    hp_text = f.read().split('\n')

prompt_list = [t[:100] if len(t) > 100 else t for t in hp_text]


def data(all_text):
    for text in all_text:
        yield text


def run():
    all_testfile = open('llama2GeneratedText_hp.json', 'a')
    cnt = 0
    for res in generator(data(prompt_list), max_new_tokens=50, batch_size=256):
        user_input = prompt_list[cnt]
        tmp = res[0]['generated_text']
        to_save = {
            'generated': tmp.lower(),
            'raw_text': user_input.lower()
        }
        json.dump(to_save, all_testfile)
        all_testfile.write("\n")
        torch.cuda.empty_cache()
        cnt += 1
    all_testfile.flush()
    all_testfile.close()

c4_path = "allenai/c4"
# load file
c4_subset = load_dataset(c4_path, data_files="en/c4-train.00001-of-01024.json.gz")['train']['text']
prompt_list2 = [t[:100] if len(t) > 100 else t for t in c4_subset]


def run2():
    all_testfile = open('llama2GeneratedText_C4.json', 'a')
    cnt = 0
    try:
        for res in tqdm.tqdm(generator(data(prompt_list2), max_new_tokens=50, batch_size=32),desc='main2:'):
            user_input = prompt_list2[cnt]
            tmp = res[0]['generated_text']
            to_save = {
                'generated': tmp.lower(),
                'raw_text': user_input.lower()
            }
            json.dump(to_save, all_testfile)
            all_testfile.write("\n")
            torch.cuda.empty_cache()
            cnt += 1
        all_testfile.flush()
        all_testfile.close()
    except Exception as e:
        # 处理 generator 异常，可以打印错误信息或采取其他措施
        print(f"Generator error: {e}")
        # 如果需要继续处理下一个输入，可以增加 cnt
        time.sleep(5)


run2()
