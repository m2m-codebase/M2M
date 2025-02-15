import pandas as pd

#df = pd.read_csv("clotho_captions_evaluation.csv")
#df = pd.read_csv("clotho_captions_development.csv")

path = "WavCaps/retrieval/data/Clotho/csv_files/train.csv"
save_p = "CLOTHO/clotho_train_from_wavcaps.csv"

df = pd.read_csv(path)

audios, captions = [], []

for i in range(len(df)):
    audio_name = df['file_name'][i][:-4]
    audios.extend([audio_name]*5)

    captions.append(df['caption_1'][i])
    captions.append(df['caption_2'][i])
    captions.append(df['caption_3'][i])
    captions.append(df['caption_4'][i])
    captions.append(df['caption_5'][i])

new_df = pd.DataFrame({"youtube_id": audios, "caption": captions})


new_df.to_csv(save_p, index=False)
print(f"Saved at: {save_p}")
