import whisper
import os, random, copy
import numpy as np
import torch
import pandas as pd
import whisper
import torchaudio
from tqdm.notebook import tqdm
import collections, json
import editdistance
from difflib import SequenceMatcher
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer
import string
from nemo_text_processing.text_normalization.normalize import Normalizer
import functools


whisper_normalizer = EnglishTextNormalizer()
nemo_normalizer = Normalizer(input_case='cased', lang='en')

DYSFLUENCIES = [
    "uh", "uhh", "uhhh", 
    "um", "umm", "ummm", 
    "mm", "mmm", "mhm", "mhmm", "mhmmm", "hm", "hmm", "hmmm", 
    "huh", "heh", "ah", "ha", "aah", "ahh", "hah",
    "lah", "lor", "meh", "wah"
]

RELAXED_PRONUNCIATIONS = {
    "gonna": ["going to"], 
    "wanna": ["want to"], 
    "dunno": ["dont know", "do not know"], 
    "gotta": ["going to"], 
    "gimme": ["give me"], 
    "kinda": ["kind of"], 
    "sorta": ["sort of"], 
    "coulda": ["could have"], 
    "shoulda": ["should have"], 
    "woulda": ["would have"], 
    "oughta": ["ought to"],
    "hafta": ["have to"],
    "ya": ["you"],
    "cuz": ["because"],
    "coz": ["because"],
    "cause": ["because"],
    "whatcha": ["what are you", "what are ya"],
    "whaddya": ["what do you", "what do ya"],
    "whadaya": ["what do you", "what do ya"],
    "whaddaya": ["what do you", "what do ya"],
    "imma": ["i am going to"]
}


def calculate_wer(pre, ref):
    return editdistance.eval(pre, ref) / len(ref)


def comp_func(a, b):
    if a['wer'] < b['wer']:
        return a
    elif a['wer'] == b['wer']:
        if a['len_differ'] <= b['len_differ']:
            return a
        else:
            return b
    else:
        return b


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model('medium.en')
tokenizer = get_tokenizer(multilingual=False)
ignore_tokens = [
    i
    for i in range(tokenizer.eot)
    if all(c in "0123456789€£" + string.punctuation for c in tokenizer.decode([i]).strip())
]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HyPost dataset generation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--asr_wav', type=str, help='wav list file')
    parser.add_argument('--asr_txt', type=str, help='transcription file')
    parser.add_argument('--hp_json', type=str, help='generated hp data file')
    parser.add_argument('--batch_size', type=int, help='inference batch size')
    parser.add_argument('--use_prompt', action='store_true')

    args = parser.parse_args()

    f_wav = open(args.asr_wav, 'r')
    f_txt = open(args.asr_txt, 'r')

    #empty_beams = 0
    json_file = []
    id = 0
    wer = 0

    prompt = None
    if args.use_prompt:
        prompt = "mhm mm hmm le let me umm lemme like hmm okay i am thi thinking umm uhh hmm i am like um ah huh and so so uh and um like um kinda like it is it is i dunno yeah okay so uh yeah so ya know it is uh uh and and uh like cuz if"
    options = whisper.DecodingOptions(language='en', beam_size=5, suppress_tokens=[-1]+ignore_tokens, length_penalty=1.0, prompt=prompt)
    lines = f_wav.readlines()
    for li in range(0, len(lines), args.batch_size):
        batch = []
        for line in lines[li: li + args.batch_size]:
            audio_path = line.strip().split()[-1]
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio) #.to(model.device)
            batch.append(mel)

        res = whisper.decode(model, torch.stack(batch).to(model.device), options)

        # print(res)

        for results in res:
            empty_beams = 0
            output = f_txt.readline().strip().split("\t")[1]
            gt = ' '.join(output.split())
            input = []
            for result in results:
                if len(input) < 5 and len(result) > 0 and result not in input:
                    input.append(result)
            if len(input) < 5:
                empty_beams = 5 - len(input)
                for _ in range(5 - len(input)):
                    repeat = copy.deepcopy(random.choice(input))
                    input.append(repeat)

            try:
                normalized_output = whisper_normalizer(nemo_normalizer.normalize(gt.strip(), verbose=False, punct_post_process=True)).strip()
            except Exception:
                normalized_output = whisper_normalizer(nemo_normalizer.normalize(gt.strip(), verbose=False, punct_post_process=True)).strip()
                print(f'normalized output exception: {normalized_output}')

            normalized_output = normalized_output if len(normalized_output) > 0 else '<UNK>'

            if args.use_prompt:
                # add dysfluencies and relaxed pronunciations back to hypotheses
                proc_input = []
                for i in input:
                    i = i.split()
                    s = SequenceMatcher(None, i, normalized_output)
                    opcodes = s.get_opcodes()
                    for op in opcodes:
                        if op[0] == 'insert':
                            if all(x in DYSFLUENCIES for x in normalized_output[op[3]:op[4]]):
                                i[op[1]:op[2]] = normalized_output[op[3]:op[4]]
                        elif op[0] == 'replace':
                            if any([
                                x in RELAXED_PRONUNCIATIONS and " ".join(i[op[1]:op[2]]) in RELAXED_PRONUNCIATIONS[x] for x in normalized_output[op[3]:op[4]]
                                ]) \
                                and all([
                                    (x in RELAXED_PRONUNCIATIONS and " ".join(i[op[1]:op[2]]) in RELAXED_PRONUNCIATIONS[x]) or (x in DYSFLUENCIES) for x in normalized_output[op[3]:op[4]]
                                ]):
                                i[op[1]:op[2]] = normalized_output[op[3]:op[4]]
                    proc_input[-1]['input'].append(" ".join(i))
                
                # filter bad hypotheses (result of Whisper's prompting)
                correct = {}
                wrong = {}
                filtered_input = []
                for ix, i in enumerate(proc_input):
                    wer = calculate_wer(i.split(), normalized_output.split())
                    len_ratio = len(i.split()) / len(normalized_output.split())
                    if wer > (2/3) or len_ratio < (2/3):
                        wrong[ix] = {
                            'hyp': i,
                            'wer': wer,
                            'len_differ': abs(1 - len_ratio)
                        }
                    else:
                        correct[ix] = {
                            'hyp': i,
                            'wer': wer,
                            'len_differ': abs(1 - len_ratio)
                        }
                    if len(correct) == 0: continue
                    best_hyp = None
                    if 0 in wrong:
                        best_hyp = functools.reduce(comp_func, list(correct.values()))
                    elif 0 in correct:
                        best_hyp = correct[0]
                    filtered_input.append({
                        'input': [best_hyp['hyp']],
                        'output': re.sub("[a-zA-Z]+- ", "", d['output']),
                        'normalized_output': d['normalized_output'],
                        'wer': best_hyp['wer']
                    })
                    for ix in range(1, 5):
                        if ix in wrong:
                            filtered_input[-1]['input'].append(random.choice(list(correct.values()))['hyp'])
                        elif ix in correct:
                            filtered_input[-1]['input'].append(correct[ix]['hyp'])
            
            cur_wer = calculate_wer(filtered_input[0].split(), normalized_output.split())
            data = {"input": filtered_input, "output": output, "normalized_output": normalized_output, "wer": cur_wer}
            print(data)
            json_file.append(data)

            # calculate wer
            id += 1
            wer += cur_wer
            print(f'Utterance {id}: WER = {cur_wer}. # Empty Beams: {empty_beams}')

    f_wav.close()
    f_txt.close()

    wer /= id
    print(f'Final WER = {wer}')

    with open(args.hp_json, 'w') as f:
        json.dump(json_file, f)

