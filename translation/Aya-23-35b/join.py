import pandas as pd
from utils import LANGS, LANGS_TO_FLORES
import os
import json

def clean_text(x):
    #print("before", x)
    if isinstance(x, list):
        x = "\n".join(x).strip()

    x = x.split("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")[-1]
    #print("after", x)
    return x

df = pd.read_csv("coco_30000_randomly_sampled_2014_val.csv")

translations = dict()

langs_to_flores = sorted([(l, l_code) for l, l_code in LANGS_TO_FLORES.items()])

col_map = dict()

for idx, (l, l_code) in enumerate(langs_to_flores):
    texts = []

    files  = df['file_name'].tolist()
    
    t_files = [
        x.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')+'.txt'
        for x in files
    ]

    save_dir = os.path.join("./gen_text", l)
    
    t_paths = [os.path.join(save_dir, f) for f in t_files]

    for fpath in t_paths:
        with open(fpath, 'r') as f:
            text = f.readlines()

        text = clean_text(text)
        texts.append(text)

    translations[f"{idx}"] = texts
    col_map[f"{idx}"] = l

df_trans = pd.DataFrame(translations)
print(df_trans)


assert len(df) == len(df_trans)

df_new = pd.concat([df, df_trans], axis=1)

save_p = "all_langs_joined.csv"
df_new.to_csv(save_p, index=False)
print(f"saved at {save_p}")

with open("lang_col_map.json", "w") as f:
    json.dump(col_map, f, indent=4)


