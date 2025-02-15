import pandas as pd
import sys
import pickle
import os

path = sys.argv[1]
df = pd.read_csv(path)
#df = df.head(20)

# audiocaps/clotho val
langs = ['en']
lang_col = ['caption']

#base_dir = 'val'
#base_dir = 'test'
base_dir = 'CLOTHO/evaluation/'

# audiocaps/clotho test

langs = [
        'eng_Latn','ben_Beng','guj_Gujr','hin_Deva',
        'kan_Knda','mal_Mlym','mar_Deva','npi_Deva',
        'pan_Guru','tam_Taml','tel_Telu','urd_Arab'
        ]
lang_col = [
        'caption','ben_Beng','guj_Gujr','hin_Deva',
        'kan_Knda','mal_Mlym','mar_Deva','npi_Deva',
        'pan_Guru','tam_Taml','tel_Telu','urd_Arab'
        ]

# aya generated translation 
# Don't use Hindi translation from AYA model
# Low spBLEU (28.2) and chrF++ (49) on flores200
# IndicTrans2 has chrF++ (59.6) on flores200
aya_cols = [
    ['eng_Latn-arb_Arab-sentence_arb_Arab', 'aya_translated_eng_Latn-arb_Arab-sentence_arb_Arab'],
    ['eng_Latn-zho_Hans-sentence_zho_Hans', 'aya_translated_eng_Latn-zho_Hans-sentence_zho_Hans'],
    ['eng_Latn-zho_Hant-sentence_zho_Hant', 'aya_translated_eng_Latn-zho_Hant-sentence_zho_Hant'],
    ['eng_Latn-ces_Latn-sentence_ces_Latn', 'aya_translated_eng_Latn-ces_Latn-sentence_ces_Latn'],
    ['eng_Latn-nld_Latn-sentence_nld_Latn', 'aya_translated_eng_Latn-nld_Latn-sentence_nld_Latn'],
    ['eng_Latn-fra_Latn-sentence_fra_Latn', 'aya_translated_eng_Latn-fra_Latn-sentence_fra_Latn'],
    ['eng_Latn-deu_Latn-sentence_deu_Latn', 'aya_translated_eng_Latn-deu_Latn-sentence_deu_Latn'],
    ['eng_Latn-ell_Grek-sentence_ell_Grek', 'aya_translated_eng_Latn-ell_Grek-sentence_ell_Grek'],
    ['eng_Latn-heb_Hebr-sentence_heb_Hebr', 'aya_translated_eng_Latn-heb_Hebr-sentence_heb_Hebr'],
    #['eng_Latn-hin_Deva-sentence_hin_Deva', 'aya_translated_eng_Latn-hin_Deva-sentence_hin_Deva'],
    ['eng_Latn-ind_Latn-sentence_ind_Latn', 'aya_translated_eng_Latn-ind_Latn-sentence_ind_Latn'],
    ['eng_Latn-ita_Latn-sentence_ita_Latn', 'aya_translated_eng_Latn-ita_Latn-sentence_ita_Latn'],
    ['eng_Latn-jpn_Jpan-sentence_jpn_Jpan', 'aya_translated_eng_Latn-jpn_Jpan-sentence_jpn_Jpan'],
    ['eng_Latn-kor_Hang-sentence_kor_Hang', 'aya_translated_eng_Latn-kor_Hang-sentence_kor_Hang'],
    ['eng_Latn-pes_Arab-sentence_pes_Arab', 'aya_translated_eng_Latn-pes_Arab-sentence_pes_Arab'],
    ['eng_Latn-pol_Latn-sentence_pol_Latn', 'aya_translated_eng_Latn-pol_Latn-sentence_pol_Latn'],
    ['eng_Latn-por_Latn-sentence_por_Latn', 'aya_translated_eng_Latn-por_Latn-sentence_por_Latn'],
    ['eng_Latn-ron_Latn-sentence_ron_Latn', 'aya_translated_eng_Latn-ron_Latn-sentence_ron_Latn'],
    ['eng_Latn-rus_Cyrl-sentence_rus_Cyrl', 'aya_translated_eng_Latn-rus_Cyrl-sentence_rus_Cyrl'],
    ['eng_Latn-spa_Latn-sentence_spa_Latn', 'aya_translated_eng_Latn-spa_Latn-sentence_spa_Latn'],
    ['eng_Latn-tur_Latn-sentence_tur_Latn', 'aya_translated_eng_Latn-tur_Latn-sentence_tur_Latn'],
    ['eng_Latn-ukr_Cyrl-sentence_ukr_Cyrl', 'aya_translated_eng_Latn-ukr_Cyrl-sentence_ukr_Cyrl'],
    ['eng_Latn-vie_Latn-sentence_vie_Latn', 'aya_translated_eng_Latn-vie_Latn-sentence_vie_Latn'],

]    

aya_langs = ["_".join(x.split("_")[-2:]) for _, x in aya_cols]
aya_lang_cols  = [x+"_cleaned" for _, x in aya_cols]

langs += aya_langs
lang_col += aya_lang_cols
    
img_col = "youtube_id"
img_files = df[img_col].tolist()
if "clotho" in path.lower():
    img_files = [x+".wav" for x in img_files]
else:
    img_files = ['Y'+x+".wav" for x in img_files]


text_to_img_mapping = {l:dict() for l in langs}
img_to_text_mapping = {l:dict() for l in langs}
SPACER = "@@@"


for c, l in zip(lang_col, langs):
    
    texts = df[c].tolist()
    
    # unique per language
    text_unique_ids = [f'caption_{i}' for i in range(len(texts))]

    unique_text_id_to_caption_map = {k:v for k,v in zip(text_unique_ids, texts)}
    text_to_img_mapping[l]["unique_text_id_to_caption_map"] = unique_text_id_to_caption_map

    for img, text_id, text in zip(img_files, text_unique_ids, texts):    
        
        img_path = os.path.join(base_dir, img)
        if not os.path.exists(img_path):
            continue
        
        if img not in img_to_text_mapping[l]:
            img_to_text_mapping[l][img] = []
        img_to_text_mapping[l][img].append(text_id)

        if text_id not in text_to_img_mapping[l]:
            text_to_img_mapping[l][text_id] = []
        text_to_img_mapping[l][text_id].append(img)



#print(text_to_img_mapping)
#print(img_to_text_mapping)

for l in langs:
    for k,v in text_to_img_mapping[l].items():
        #text_to_img_mapping[l][k] = list(set(v))
        print(len(text_to_img_mapping[l][k]), end=" ")

        if k != "unique_text_id_to_caption_map":
            assert len(text_to_img_mapping[l][k]) == 1 , f"{len(text_to_img_mapping[l][k])} text to image"
    print()
    print('-'*100)
    for k,v in img_to_text_mapping[l].items():
        #img_to_text_mapping[l][k] = list(set(v))
        print(len(img_to_text_mapping[l][k]), end=" ")
        assert len(img_to_text_mapping[l][k]) == 5
    print()
    print(l)
    print('-'*100)

save1 = path.replace(".csv", "_text_to_img_map.pkl")
with open(save1, "wb") as f:
    pickle.dump(text_to_img_mapping, f)

save2 = path.replace(".csv", "_img_to_text_map.pkl")
with open(save2, "wb") as f:
    pickle.dump(img_to_text_mapping, f)


print(f"File saved at {save1} and {save2}")
