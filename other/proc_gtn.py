import os
import json
import pandas as pd
from mosestokenizer import MosesDetokenizer
import random
from tqdm import tqdm
from ..generate_data.whisper.whisper.normalizers import EnglishTextNormalizer
from nemo_text_processing.text_normalization.normalize import Normalizer


whisper_normalizer = EnglishTextNormalizer()
nemo_normalizer = Normalizer(input_case='cased', lang='en')
detokenize = MosesDetokenizer('en')

gtn_dir = "<PATH_TO_GTN>"
gtn = pd.read_csv(os.path.join(gtn_dir, "output_1.csv"), nrows=8000000)

print(gtn)

gtn_json = {}


count = 0
itn_sent = []
tn_sent = []
i = 0
while i <= len(gtn):
    if gtn.iloc[i]["Input Token"] == "<eos>":
        if len(itn_sent) == 0 or len(tn_sent) == 0: 
            itn_sent = []
            tn_sent = []
            i += 1
            continue
        text = " ".join(tn_sent)
        if text.istitle():
            itn_sent = []
            tn_sent = []
            i += 1
            continue
        if itn_sent[0] == 'Retrieved' or (itn_sent[0] == '"' and (itn_sent[-1] == '"' or itn_sent[-2] == '"')):
            itn_sent = []
            tn_sent = []
            i += 1
            continue
        if len(text) <= 6:
            itn_sent = []
            tn_sent = []
            i += 1
            continue
        if 'sil' in text or '_letter' in text:
            itn_sent = []
            tn_sent = []
            i += 1
            continue
        text = " ".join(itn_sent)
        if " ".join(itn_sent).count('&') > 3 or " ".join(tn_sent).count(',') > 5:
            itn_sent = []
            tn_sent = []
            i += 1
            continue
        if '...' in text or '|' in text:
            itn_sent = []
            tn_sent = []
            i += 1
            continue
        if sum(char.isdigit() for char in text) > len(text)/2:
            itn_sent = []
            tn_sent = []
            i += 1
            continue
        if sum(len(char.encode()) > 1 for char in text) > len(text)/4:
            itn_sent = []
            tn_sent = []
            i += 1
            continue
        itn_sent =  detokenize(itn_sent).strip()  
        tn_sent = " ".join(tn_sent)
        tn_sent = whisper_normalizer(
            nemo_normalizer.normalize(tn_sent.strip(), verbose=False, punct_post_process=True)
        ).strip()
        gtn_json.append({ "input": [tn_sent], "output": itn_sent })
        itn_sent = []
        tn_sent = []
        count += 1
        if count % 10000 == 0: print(i, count)
        if count >= 400000:
            count = 0
            break
        i += 1
        continue
    
    if gtn.iloc[i]["Semiotic Class"] != "VERBATIM":
        itn_sent.append(str(gtn.iloc[i]["Input Token"]))
        if gtn.iloc[i]["Output Token"] == "<self>":
            tn_sent.append(str(gtn.iloc[i]["Input Token"]))
        elif gtn.iloc[i]["Output Token"] != "sil":
            if gtn.iloc[i]["Semiotic Class"] == "LETTERS":
                gtn.iloc[i]["Output Token"] = "".join(gtn.iloc[i]["Output Token"].split())
            tn_sent.append(str(gtn.iloc[i]["Output Token"]))
    i += 1

print("Number of Sentences:", count)


with open(os.path.join(gtn_dir, "gtn_baseline_train.json"), "w") as f:
    json.dump(gtn_json, f)
