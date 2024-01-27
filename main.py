# 使用pipeline加载模型
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_path = "microsoft/Llama2-7b-WhoIsHarryPotter"
tokenizer_path = "microsoft/Llama2-7b-WhoIsHarryPotter"
torch.cuda.empty_cache()
generator = pipeline("text-generation",
                     model=model_path,
                     tokenizer=model_path,
                     device_map="auto",
                     batch_size=4)
print("model loaded")
tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)


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
import json

file_path = 'hp_score_true.txt'
# read file to var hp_text
with open(file_path, 'r', encoding='utf-8') as f:
    hp_text = f.read().split('\n')

prompt_list = [t[:100] if len(t) > 100 else t for t in hp_text]


def run():
    all_testfile = open('llama2GeneratedText.json', 'a')
    for prompt in prompt_list:
        # 调用llama生成下一句话
        user_input = prompt
        text = user_input
        tmp = generator(text,
                        max_new_tokens=30,
                        # do_sample=True,
                        )[0]['generated_text']
        # 将生成的下一句话tmp和删除的词对比，如果tmp和删除的词一样，那么就把prompt作为一个列表存入文件testfile中
        to_save = {
            'generated': tmp.lower(),
            'raw_text': user_input.lower()
        }
        json.dump(to_save, all_testfile)
        all_testfile.write("\n")
        torch.cuda.empty_cache()
    all_testfile.flush()
    all_testfile.close()


run()
