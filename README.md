# HyPost: Effective General and Domain-Adaptable ASR Post-Processing using LLMs

Submission to ICASSP 2025

## Usage

```bash
pip install -r requirements.txt

```

### Generating the HyPost Dataset
```bash
cd generate_data/whisper
pip install -e .

python generate_hypost_dataset.py --asr_wav  --asr_wav /path/to/wav --asr_txt /path/to/text --hp_json /path/to/hp.json --use_prompt
```

- `asr_wav`: list of utterance ids and paths, e.g. "utt_id_1 /path/to/1.wav";
- `asr_txt`: list of utterance ids and transcripts e.g. "utt_id_1 i have a dream";
- `hp_json`: generated json file containing 5 hypotheses (`input`), unnormalized ground truth transcript (`output`), ground-truth transcript after normalization (`normalized_output`), and WER using normalized transcript (`wer`);
- `use_prompt`: prompt to include hesitations in the hypotheses

### LoRA Training
```bash
python finetune.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path './data/hypost.json' \
    --output_dir './hypost' \
    --lora_target_modules='["down_proj","gate_proj","up_proj"]' \
    --learning_rate 1e-4 \
    --micro_batch_size=64 \
    --batch_size=256 \
    --lora_r=16 \
    --lora_alpha=16 \
    --prompt_template_name 'HyPost-LoRA' # change for other tasks in finetune/templates 
```


Adapted from https://github.com/Hypotheses-Paradise/Hypo2Trans and https://github.com/tloen/alpaca-lora