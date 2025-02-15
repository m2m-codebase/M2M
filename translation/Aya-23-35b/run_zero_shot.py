import torch
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



en_col = "sentence_eng_Latn"
en_col2 = "caption"

csv_path = sys.argv[1]
prompt_file = sys.argv[2]
sampling_csv_path = sys.argv[3]
nsamples = int(sys.argv[4])
batch_size = int(sys.argv[5])
MAX_TOKENS=128
feat_dir = "./labse_features"


# ------------------ This is for sampling few shots -------------------------------
# generate labse features and pkl files separately, fore running this file



# Compute few shots for all text
# Re-use for each En->X translation
df = pd.read_csv(csv_path)#.head(3)


#----------------------------------------- PROMPT ------------------------------------------------------
with open(prompt_file, "r") as f:
    prompt = f.readlines()
prompt = "\n".join([x.strip() for x in prompt])

# translate in AYA supported languages
# use flores code to retrieve few-shots

lang_to_flores_ = [(l, l_code) for l, l_code in LANGS_TO_FLORES.items()]

# reverse
#lang_to_flores_ = lang_to_flores_[::-1]

for lang, flores_code in lang_to_flores_:
    
    col = f"eng_Latn-{flores_code}-sentence_{flores_code}" 
    all_prompts = []

    prompt_lang = prompt.replace("{tgt_lang}", lang)

    en_texts = df[en_col2].tolist()
    img_names = df['file_name'].tolist()

    translated_texts = []
    save_paths = [
            x.replace('.jpg', f'').replace('.png', f'').replace('.jpeg', '')+'.txt'
            for x in img_names
        ]

    save_dir = os.path.join("./gen_text", lang)
    os.makedirs(save_dir, exist_ok=True)

    save_paths = [os.path.join(save_dir, x) for x in save_paths]

    for en_text in tqdm(en_texts, desc=f"En -> {lang}"):

        prompt_lang_few_shot = prompt_lang.replace("{source_text}", en_text)
        
        all_prompts.append(prompt_lang_few_shot)

        # print(prompt_lang_few_shot)
        # print("*"*100)

    get_translations(all_prompts, save_paths, batch_size)
    '''
    col = "aya_translated_"+col

    # -------------------- call translation function -----------------------------
    translated_texts_dict = dict()
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
    lang_name = lang.replace(' ', '_').strip()
    save_p = csv_path.replace('.csv', f'all_{prompt_name}_nsamples{nsamples}_{lang_name}.csv')
    df_new.to_csv(save_p, index=False)
    print(f"File saved at: {save_p}")
    '''

