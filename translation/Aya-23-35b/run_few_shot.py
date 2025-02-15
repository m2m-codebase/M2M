import torch
from sentence_transformers import SentenceTransformer
import sys
import pickle
import pandas as pd
from tqdm import tqdm
import os

from utils import LANGS, LANGS_TO_FLORES

from aya_model import get_translations

def get_labse_feature(labse_model, text):
    # should return 1xd feature
    text_embedding = labse_model.encode([text])
    text_embedding = torch.from_numpy(text_embedding)

    return text_embedding

def sample_few_shots(text, sample_feats, nsamples, labse_model, text_idx=-1):
   
    if nsamples == 0: return []

    text_feat = get_labse_feature(labse_model, text).to('cuda')

    # 1 x d @ d x N -> 1 x N
    sims = text_feat @ sample_feats.permute(1,0)
    #print("sims.shape", sims.shape)
   
    # sample 1 extra
    # in case new text is identical to few-shot sample
    # only for flores eval
    topk_values, topk_indices = torch.topk(sims, nsamples+1)
    
    #print("topk v", topk_values)
    #print("topk i", topk_indices)

    # TODO implement merged logic for dev+devtest

    few_shot_idxs = topk_indices.view(-1).tolist()
    few_shot_idxs = [x for x in few_shot_idxs if x!=text_idx]

    return few_shot_idxs[:nsamples]


LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
labse_model = SentenceTransformer(LABSE_MODEL_NAME).to('cuda')

en_col = "sentence_eng_Latn"
en_col2 = "caption"

csv_path = sys.argv[1]
prompt_file = sys.argv[2]
sampling_csv_path = sys.argv[3]
nsamples = int(sys.argv[4])
MAX_TOKENS=128
feat_dir = "./labse_features"


# ------------------ This is for sampling few shots -------------------------------
# generate labse features and pkl files separately, fore running this file

# data for sampling few-shots
df_test = pd.read_csv(sampling_csv_path)
sample_texts = df_test[en_col]
sample_texts = list(set(sample_texts))

text2id_path = "flores_text2id.pkl"
with open(text2id_path, "rb") as f:
    text2id = pickle.load(f)

sample_feats = []
text_order = dict()
for idx, t in tqdm(enumerate(sample_texts), desc="Loading few-shot feats"):
    text_id = text2id[t]
    text_order[t] = idx
    feat_path = os.path.join(feat_dir, f"{text_id:05d}.pt")
    feat = torch.load(feat_path)
    sample_feats.append(feat)

# N x d
sample_feats = torch.cat(sample_feats, dim=0).to('cuda')
print("sample_feats", sample_feats.shape)

# Compute few shots for all text
# Re-use for each En->X translation
text2few_shot = dict()
df = pd.read_csv(csv_path)
test_text = df[en_col2].tolist()

for t in tqdm(test_text, desc="computing few-shots"):
    if t not in text2few_shot:
        few_shot_idxs = sample_few_shots(
            t, sample_feats, 
            nsamples=nsamples, 
            labse_model=labse_model, 
            text_idx = -1
        )
        text2few_shot[t] = few_shot_idxs



#----------------------------------------- PROMPT ------------------------------------------------------
with open(prompt_file, "r") as f:
    prompt = f.readlines()
prompt = "\n".join([x.strip() for x in prompt])

# translate in AYA supported languages
# use flores code to retrieve few-shots
translated_texts_dict = dict()

for lang, flores_code in LANGS_TO_FLORES.items():
    col = f"eng_Latn-{flores_code}-sentence_{flores_code}" 
    all_prompts = []

    prompt_lang = prompt.replace("{tgt_lang}", lang)

    en_texts = df[en_col2].tolist()

    translated_texts = []

    for en_text in tqdm(en_texts, desc=f"En -> {lang}"):
        few_shot_egs = []
        few_shot_idxs = text2few_shot[en_text]

        for i_, idx in enumerate(few_shot_idxs, 1):
            # print("idx", idx)
            eg_en = df_test[en_col][idx]
            eg_tgt = df_test[col][idx]

            eg = f"""Example {i_}:\nInput text: {eg_en}\nTranslation: {eg_tgt}"""
            few_shot_egs.append(eg)

        few_shot_prompt = "\n".join(few_shot_egs)

        prompt_lang_few_shot = prompt_lang.replace("{fewshots}", few_shot_prompt)
        prompt_lang_few_shot = prompt_lang_few_shot.replace("{source_text}", en_text)
        
        all_prompts.append(prompt_lang_few_shot)

        # print(prompt_lang_few_shot)
        # print("*"*100)

    col = "aya_translated_"+col

    # -------------------- call translation function -----------------------------
    translated_texts = get_translations(all_prompts)
    translated_texts_dict[col] = translated_texts
    translated_texts_dict[col+"_prompts"] = all_prompts
    print("="*100)

# -------------------------------- Saving --------------------------------------------------------------------
trans_df = pd.DataFrame(translated_texts_dict).reset_index(drop=True)
df = df.reset_index(drop=True)

assert len(df) == len(trans_df), f"df shapes dont match, {df.shape} and {trans_df.shape}"

df_new = pd.concat([df, trans_df], axis=1)

prompt_name = prompt_file.split('/')[-1].replace('.txt', '')
save_p = csv_path.replace('.csv', f'{prompt_name}_nsamples{nsamples}_testing.csv')
df_new.to_csv(save_p, index=False)
print(f"File saved at: {save_p}")

