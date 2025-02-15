import json
import pandas as pd
import pickle

#file_path = "../eval_data/crossmodal3600/captions.jsonl"
file_path = "captions.jsonl"

all_data = []

langs = [
'ar', 'bn', 'cs', 'da', 'de', 'el', 'en',
       'es', 'fa', 'fi', 'fil', 'fr', 'hi', 'hr', 'hu', 'id', 'it', 'he', 'ja',
              'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'quz', 'ro', 'ru', 'sv', 'sw', 'te',
                     'th', 'tr', 'uk', 'vi', 'zh',
]

c = 0

# all are per lang

text_to_image = {l: dict() for l in langs}
image_to_text = {l: dict() for l in langs}

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())

        all_data.append(data)

        image_name = data['image/key']
        for l in langs:

            if len(data[l]['caption']) != len(set(data[l]['caption'])):
                print(data[l]['caption'])
            
            data[l]['caption'] = list(set(data[l]['caption']))

            for text in data[l]['caption']:
                if text not in text_to_image[l]:
                    text_to_image[l][text] = []
                
                #if image_name in text_to_image[l][text]:
                #    print("dup img", image_name, text)
                
                text_to_image[l][text] += [image_name]
            
                if image_name not in image_to_text[l]:
                    image_to_text[l][image_name] = []

                #if text in image_to_text[l][image_name]:
                #    print("dup text", image_name, text)
                
                image_to_text[l][image_name] += [text]
            
# total images
len_t2i = {l: sum([len(x) for x in v.values()]) for l,v in text_to_image.items()}
# unique images per text
unique_len_t2i = {l: sum([len(set(x)) for x in v.values()]) for l,v in text_to_image.items()}

# ideally both should be same
# no duplicates within (lang,text) key
print(len_t2i)
print(unique_len_t2i)
print(len_t2i == unique_len_t2i)

with open("../eval_data/crossmodal3600/text_to_image_mapping.pkl", "wb") as f:
    pickle.dump(text_to_image, f)
with open("../eval_data/crossmodal3600/image_to_text_mapping.pkl", "wb") as f:
    pickle.dump(image_to_text, f)

print("Saved at", "../eval_data/crossmodal3600/text_to_image_mapping.pkl")
print("Saved at", "../eval_data/crossmodal3600/image_to_text_mapping.pkl")
exit()

caption_count_per_lang = {}
print(caption_count_per_lang)
print(sum([v for v in caption_count_per_lang.values()]))
unique_cap = {l: len(set(cap)) for l, cap in captions.items()}
print(unique_cap)
print(sum(unique_cap.values()))

image_count = {t: len(set(v)) for t, v in text_to_image.items()}
# more than 1 image for a text is found
print(sum([v-1 for v in image_count.values() if v>1]))
exit()

print(len(all_data))
print(all_data[0].keys())
print(c/len(langs))
df = pd.DataFrame(data)
print(df)
print(df.columns)
print(df.shape)


