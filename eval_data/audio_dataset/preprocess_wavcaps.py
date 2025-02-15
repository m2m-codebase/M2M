import json
import random

texts = []
exclude_texts = []

paths = [
    "WavCaps/data/json_files/SoundBible/sb_final.json",
    "WavCaps/data/json_files/AudioSet_SL/as_final.json",
    "WavCaps/data/json_files/BBC_Sound_Effects/bbc_final.json",
    "WavCaps/fsd_final.json",
    "WavCaps/fsd_final_2s.json",
]

exclude_paths = [
    "WavCaps/data/json_files/blacklist/blacklist_exclude_all_ac.json",       
    "WavCaps/data/json_files/blacklist/blacklist_exclude_test_ac.json",
    "WavCaps/data/json_files/blacklist/blacklist_exclude_ub8k_esc50_vggsound.json",
]
audio_ids = {
    "AudioSet": [],
    "FreeSound": [],
}
for path in exclude_paths:
    print(path)
    with open(path, 'r') as f:
        data = json.load(f)

    audio_ids['AudioSet'] += data['AudioSet']
    audio_ids['FreeSound'] += data['FreeSound']

audio_ids = {k:set(v) for k,v in audio_ids.items()}

removed_audios = 0
for path in paths:
    print(path)
    with open(path, 'r') as f:
        data = json.load(f)


    for sample in data['data']:
        if "audioset" in path.lower():
            if sample['id'] in audio_ids['AudioSet']:
                removed_audios += 1
                continue

        if "fsd" in path.lower():
            if sample['id'] in audio_ids['FreeSound']:
                removed_audios += 1
                continue

        captions = sample['caption']
        if isinstance(captions, str):
            captions = [captions]

        texts += captions

print(len(texts), len(set(texts)))
texts = list(set(texts))

texts = [x for x in texts if x not in exclude_texts]
print(len(texts))
print("removed audios", removed_audios, len(audio_ids['AudioSet']), len(audio_ids['FreeSound']))

random.shuffle(texts)

save_p = "wavcaps_train.txt"
with open(save_p, "w") as f:
    f.write("\n".join(texts))

print(f"Saved at: {save_p}")
