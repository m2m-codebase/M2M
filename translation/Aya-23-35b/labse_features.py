import torch
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm
import pickle

def create_embed(labse_model, feat_save_path):
    if not os.path.exists(feat_save_path):
        text_embedding = labse_model.encode([text])
        text_embedding = torch.from_numpy(text_embedding)
        
        torch.save(text_embedding, feat_save_path)
        return "done"
    return "already exists"


LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
labse_model = SentenceTransformer(LABSE_MODEL_NAME).to('cuda')

csv_file = "flores200_test_dev_merged.csv"
save_dir = "labse_features"
df = pd.read_csv(csv_file)
en_col = "sentence_eng_Latn" 

# assign unique ID to each text
text2id_path = "flores_text2id.pkl"
if os.path.exists(text2id_path):
    with open(text2id_path, "rb") as f:
        text2id = pickle.load(f)
else:
    text2id = dict()

for c in [en_col]:
    text = df[c].tolist()
    for t in text:
        if t not in text2id:
            text2id[t] = len(text2id)

# make features

for text, ID in tqdm(text2id.items(), desc="Processing"):
    print(text)
    save_path = os.path.join(save_dir, f"{ID:05d}.pt")
    print(save_path)
    status = create_embed(labse_model, save_path)
    
    print(status)
    print("*")

with open(text2id_path, "wb") as f:
    pickle.dump(text2id, f)


print(f"Saved at: {text2id_path}")
