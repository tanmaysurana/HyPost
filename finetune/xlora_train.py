import torch
import fire
import xlora
import os
import sys
from typing import List, Union
from transformers import T5Tokenizer, AutoConfig, T5ForConditionalGeneration, TrainingArguments, DataCollatorForSeq2Seq, Trainer,  AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utils.prompter import Prompter
from trl import SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_token = "<HF_TOKEN>"

from huggingface_hub import login
login(token=hf_token)

template =  {
    "description": "HyPost template used by XLoRA tuning.",
    "prompt_hypost": "Below is the best-hypothesis transcribed from a speech recognition system. Please try to revise it using only the words included in other-hypotheses, and write the response for the true transcription. Furthermore, please generate the true transcription in a readable form by converting the appropriate words to numbers and symbols, adding punctuations where necessary, and identifying sentence boundaries (note that the transcript maybe an incomplete sentence or paragraph, missing its beginning and/or end).\n\n### Best-hypothesis:\n{best}\n\n### Other-hypotheses:\n{others}\n\n### Response:\n",
    "response_split": "### Response:"
}

def train(
    # model/data params
    data_path: str = "./combined_data.json", # required
    adapters_path: List[str] = [
        "<PATH_TO_HYPOST_LORA>",
        "<PATH_TO_IN_DOMAIN_LORA>",
    ],
    remove_columns: List[str] = [
        "input",
        "output"
    ],
    output_dir: str = '', # required
    base_model: str = "meta-llama/Llama-2-7b-hf",
    # training hyperparams
    batch_size: int = 4,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 5e-5,
    cutoff_len: int = 768,
    val_set_size: int = 2048,
    #val_set_size: int = 1024,

    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter

    prompt_template_name: str = "HyPost-LoRA"
):
    gradient_accumulation_steps = batch_size // micro_batch_size
   
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1 ))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    prompter = Prompter(prompt_template_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        use_flash_attention_2=False,
        token=hf_token
    )

    config = AutoConfig.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_flash_attention_2=False,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left', token=hf_token)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference


    data_collator = DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True, pad_to_multiple_of=8, model=model
    )

    def generate_prompt(
        best: Union[None, str] = None,
        others: Union[None, str] = None,
        label: Union[None, str] = None,    
    ) -> str:
        
        if others is None:
            res = template["prompt_hypost"].format(best = best)
        if others is not None:
            res = template["prompt_hypost"].format(best = best, others = others)

        if label:
            res = f"{res}{label}"

        return res

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            input=data_point["input"][0],
            label=data_point["response"],
            input2=data_point["input"][1:],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        return tokenized_full_prompt

    model.config.use_cache = False
    print(adapters_path[0])
    print(adapters_path[1])
    ### Convert the model to X-LoRA
    model_created = xlora.add_xlora_to_model(
    model=model,
    xlora_config=xlora.xLoRAConfig(
        model.config.hidden_size,
        base_model_id=base_model,
        xlora_depth=4,
        device=torch.device("cuda"),
        adapters={
            "adapter_1": adapters_path[0],
            "adapter_2": adapters_path[1],
        },
        use_trainable_adapters = False,
        enable_softmax = True,
        enable_softmax_topk = False,
        layerwise_scalings = True,
        enable_relu_and_dropout = True,
        use_bias = True,
        xlora_dropout_p = 0.2,
        softmax_temperature = 2,
        scaling_pass_value = 0,
        global_scaling_weight = 1,
    ),
    verbose=True,
    )

    
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "xlora_classifier.safetensors"
            )  # only xLoRA model - xLoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        #The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            print(torch.cuda.get_device_name(0))
            model_created = xlora.from_pretrained(checkpoint_name, model_created, "cuda",)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model_created.print_trainable_parameters()


    if val_set_size > 0:
        train_val = data["train"].shuffle().train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=remove_columns)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=remove_columns)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=remove_columns)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model_created.is_parallelizable = True
        model_created.model_parallel = True

    args = TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=500,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            logging_steps=10,
            optim="paged_adamw_8bit",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=500 if val_set_size > 0 else None,
            save_steps=5000,
            max_grad_norm=0.3,
            output_dir=output_dir,
            #save_total_limit=5,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to= None,
            )
    


    trainer = Trainer(
        model = model_created,
        train_dataset=train_data,
        eval_dataset=val_data,
        args = args,
        data_collator= data_collator,
        tokenizer=tokenizer,    
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model_created = torch.compile(model_created)

    trainer.train()

    model_created.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)

