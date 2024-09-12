from typing import Literal

import fire
import torch
import transformers
from datasets import load_dataset

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from h2t_lora.utils.prompter import Prompter

from tqdm import tqdm

import json

import editdistance
from generate_data.whisper.whisper.normalizers import EnglishTextNormalizer
from nemo_text_processing.text_normalization.normalize import Normalizer
from nltk.tokenize import RegexpTokenizer


whisper_normalizer = EnglishTextNormalizer()
nemo_normalizer = Normalizer(input_case='cased', lang='en')
wordpunct_tokenizer = RegexpTokenizer(r'[a-zA-Z]+|[0-9.,]+|[^\w\s]+')


def eval(
    # model/data params
    lora_path: str = "",
    data_path: str = "", # required
    base_model: str = "meta-llama/Llama-2-7b-hf",
    # training hyperparams
    batch_size: int = 64,
    cutoff_len: int = 1024,
    prompt_template_name: str = "HyPost-LoRA",  # The prompt template to use
    spoken_form_output: bool = False, # set to true for hyporadise
    input_type: Literal['multihyp', 'singlehyp', 'rephyp'] = 'multihyp', # multihyp: multiple hypotheses (HyPost), singlehyp: single hypothesis (GTN), rephyp: one hypothesis repeated 5 times
    output_path: str = "" # required
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template_name)

    device_map = "auto"

    model_ = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        token="<HF_TOKEN>"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, token="<HF_TOKEN>")

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

   
    def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None
        )

        return result


    def generate_and_tokenize_prompt(data_point):
        input2 = None
        if input_type == 'multihyp':
            input2 = data_point["input"][1:]
        if input_type == 'singlehyp':
            input2 = None
        if input_type == 'rephyp':
            input2 = [data_point["input"][0]]*4
        

        prompt = prompter.generate_prompt(
            input=data_point["input"][0],
            input2=input2,
        )

        tokenized_full_prompt = tokenize(prompt)

        return tokenized_full_prompt
    

    model = PeftModel.from_pretrained(model_, lora_path, adapter_name="lora_1")
    model.eval()

    with open(data_path) as f:
        data = json.load(f)
    eval_prompts = [generate_and_tokenize_prompt(d) for d in data] 


    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


    def calculate_wer(pre, ref):
        return editdistance.eval(pre, ref) / len(ref)


    best_wers = []
    output_file = open(output_path, "w")
    def get_ter_report(batch, res):
        for i, r in zip(batch, res):
            try:
                r = prompter.get_response(r)
                r = r.strip().split("\n")[0].strip()
            except:
                r = "<unk>"
            output_file.write(r + "\n") 
            normalized_r = nemo_normalizer.normalize(r, verbose=False, punct_post_process=True)
            normalized_r = whisper_normalizer(normalized_r).strip()
            wer = calculate_wer(normalized_r.split(), i['normalized_output'].split())
            ter = 0.0
            if not spoken_form_output:
                ter = calculate_wer(wordpunct_tokenizer.tokenize(r.strip()), wordpunct_tokenizer.tokenize(i['output']))
            best_wers.append({ "best_hyp_wer": i['ter'], "wer": wer, "ter": ter })


    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_prompts), batch_size)):
            batch = eval_prompts[i:i+batch_size]
            tokenized_batch = data_collator(batch)
            tokenized_batch = tokenized_batch['input_ids'].to("cuda")
            outputs = model.generate(input_ids=tokenized_batch, max_new_tokens=256, pad_token_id=0, do_sample=False, top_p=None)
            res = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
            get_ter_report(data[i:i+batch_size], res)


    print( "Best Hypothesis WER (Normalized):", sum([x["best_hyp_wer"] for x in best_wers]) / len(best_wers) )
    print( "WER (Normalized):", sum([x["wer"] for x in best_wers]) / len(best_wers) )
    print( "TER (Denormalized):", sum([x["ter"] for x in best_wers]) / len(best_wers) )


if __name__ == "__main__":
    fire.Fire(eval)
