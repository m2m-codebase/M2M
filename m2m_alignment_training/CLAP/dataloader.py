import torch
import os
import pandas as pd
import pickle
from PIL import Image
import requests
import librosa

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, eng_file_paths, non_eng_file_paths,
                labse_model=None, clip_model=None, clip_processor=None, 
                clip_text_to_id_path=None, labse_text_to_id_path=None,
                feature_save_dir=None, audio_model = 'clip', text_model = 'labse',
        ):
        # NOTE: for parallel data
        # make sure eng_file_paths and non_eng_file_paths
        # are aligned

        self.eng_file_paths = eng_file_paths
        self.non_eng_file_paths = non_eng_file_paths

        # TODO:
        # currently assumes len(eng_data) == len(non_eng_data)
        # one can have diff size for both
        # but then batching becomes tricky to do
        # think on it later
        self.eng_data = []
        self.non_eng_data = []

        self.labse_model = labse_model
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        self.device = clip_model.device

        
        self.audio_model = audio_model
        self.text_model = text_model

        self.gather_data()

        self.feature_save_dir = feature_save_dir
        self.clip_text_to_id_path = clip_text_to_id_path
        self.labse_text_to_id_path = labse_text_to_id_path

        self.clip_text_to_id = self.create_text_to_id(
                self.clip_text_to_id_path, self.eng_data
        )
        self.labse_text_to_id = self.create_text_to_id(
                self.labse_text_to_id_path, self.eng_data+self.non_eng_data
        )


        with open(self.clip_text_to_id_path, 'wb') as f:
            pickle.dump(self.clip_text_to_id, f)
        
        with open(self.labse_text_to_id_path, 'wb') as f:
            pickle.dump(self.labse_text_to_id, f)

    def gather_data(self):
        # extract data from text files
        # from base_paths and save in data

        # eng data
        for file_path in self.eng_file_paths:
            with open(file_path, 'r') as f:
                eng_lines = f.readlines()

            eng_lines = [l.strip() for l in eng_lines]
            self.eng_data.extend(eng_lines)

        # non eng data
        for file_path in self.non_eng_file_paths:
            with open(file_path, 'r') as f:
                non_eng_lines = f.readlines()

            non_eng_lines = [l.strip() for l in non_eng_lines]
            self.non_eng_data.extend(non_eng_lines)
    
    def create_text_to_id(self, text_to_id_path, text_data):
        text_to_id = dict()
        if os.path.exists(text_to_id_path):
            with open(text_to_id_path, 'rb') as f:
                text_to_id = pickle.load(f)
        
        # add all new texts
        for text in text_data:
            if text not in text_to_id:
                text_to_id[text] = len(text_to_id)

        return text_to_id

    def __len__(self):
        return len(self.eng_data)

    def __getitem__(self, idx):
        eng_text = self.eng_data[idx]
        non_eng_text = self.non_eng_data[idx] if self.non_eng_data else None

        eng_clip_emb = None
        eng_labse_emb = None
        non_eng_labse_emb = None

        if self.clip_model is not None:
            eng_text_clip_id = self.clip_text_to_id[eng_text]
            eng_text_clip_emb_path = os.path.join(
                    self.feature_save_dir, "train", self.audio_model, f"{eng_text_clip_id}.pth"           ) 
            os.makedirs(os.path.dirname(eng_text_clip_emb_path), exist_ok=True)
            if os.path.exists(eng_text_clip_emb_path):
                eng_clip_emb = torch.load(eng_text_clip_emb_path)
            else:
                inputs = self.clip_processor(
                    text=[eng_text], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                      inputs[k] = v.to(self.device)

                eng_clip_emb = self.clip_model.get_text_features(**inputs).detach().cpu()
                torch.save(eng_clip_emb, eng_text_clip_emb_path)

        if self.labse_model is not None:
            
            eng_text_labse_id = self.labse_text_to_id[eng_text]
            eng_text_labse_emb_path = os.path.join(
                    self.feature_save_dir, "train", self.text_model, 
                    f"{eng_text_labse_id}.pth"
            )
            os.makedirs(os.path.dirname(eng_text_labse_emb_path), exist_ok=True)
            if os.path.exists(eng_text_labse_emb_path):
                eng_labse_emb = torch.load(eng_text_labse_emb_path)
            else:
                eng_labse_emb = torch.from_numpy(self.labse_model.encode([eng_text]))
                torch.save(eng_labse_emb, eng_text_labse_emb_path)
            
           
            if non_eng_text is not None:
                non_eng_text_labse_id = self.labse_text_to_id[non_eng_text]
                non_eng_text_labse_emb_path = os.path.join(
                        self.feature_save_dir, "train", self.text_model, 
                        f"{non_eng_text_labse_id}.pth"
                )
                if os.path.exists(non_eng_text_labse_emb_path):
                    non_eng_labse_emb = torch.load(non_eng_text_labse_emb_path)
                else:
                    non_eng_labse_emb = torch.from_numpy(self.labse_model.encode([non_eng_text]))
                    torch.save(non_eng_labse_emb, non_eng_text_labse_emb_path)

        return {
            'eng_clip_emb': eng_clip_emb,
            'eng_labse_emb': eng_labse_emb,
            'non_eng_labse_emb': non_eng_labse_emb if non_eng_labse_emb is not None else [],
        }


"""## Evaluation Dataset"""

# Initialize CLIP model and processor
class EmbeddingPrecomputator:
    def __init__(self, clip_model, clip_processor, labse_model, device,
                 audio_dir, feature_save_dir, img_to_text_map_path, text_to_img_map_path, 
                 text_to_id_path = None,
                audio_model = "clip", text_model = "labse", data_split="",
        ):
        self.processor = clip_processor
        self.model = clip_model
        self.device = device
        self.labse_model = labse_model
        self.img_to_text_map_path = img_to_text_map_path
        self.text_to_img_map_path = text_to_img_map_path
        self.audio_dir = audio_dir
        self.feature_save_dir = feature_save_dir
        self.text_to_id_path = text_to_id_path
        self.audio_model = audio_model
        self.text_model = text_model
        self.data_split = data_split

        self.audio_feat_dir = f"audios_{self.audio_model}"
        
        if os.path.exists(text_to_id_path):
            with open(text_to_id_path, 'rb') as f:
                self.text_to_id = pickle.load(f)
        else:
            self.text_to_id = dict()
        
        
        with open(self.img_to_text_map_path, "rb") as f:
            self.img_to_text_map = pickle.load(f)
        with open(self.text_to_img_map_path, "rb") as f:
            self.text_to_img_map = pickle.load(f)

        self.create_text_to_id()

        with open(text_to_id_path, 'wb') as f:
            pickle.dump(self.text_to_id, f)

        
    def create_text_to_id(self):
        langs = sorted(self.img_to_text_map.keys())
        

        for l in langs:
        
            unique_text_id_to_caption_map = dict()
            if "unique_text_id_to_caption_map" in self.text_to_img_map[l]:
                unique_text_id_to_caption_map = self.text_to_img_map[l].get("unique_text_id_to_caption_map")

            print(f"{l}- unique_text_id_to_caption_map", len(unique_text_id_to_caption_map))
        
            for text_ in unique_text_id_to_caption_map.values():
                if text_ not in self.text_to_id:
                    self.text_to_id[text_] = len(self.text_to_id)
                
            '''
            for k, v in self.img_to_text_map[l].items():
                if isinstance(v, str):
                    v = [v]

                for text_ in v:
                    if text_ not in self.text_to_id:
                        self.text_to_id[text_] = len(self.text_to_id)
            
            for v, k in self.text_to_img_map[l].items():
                if isinstance(v, str):
                    v = [v]

                for text_ in v:
                    if text_ not in self.text_to_id:
                        self.text_to_id[text_] = len(self.text_to_id)
            '''

    @torch.no_grad()
    def get_audio_embedding(self, audio_name):
        aud_name_no_ext = audio_name.replace(".wav", "")
        feat_save_path = os.path.join(self.feature_save_dir, self.data_split, self.audio_feat_dir, aud_name_no_ext + '.pt')
        os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)

        if os.path.exists(feat_save_path):
            audio_embedding = torch.load(feat_save_path)
        else:
            audio_path = os.path.join(self.audio_dir, audio_name)
            audio_sample, sr = librosa.load(audio_path, sr=None)  # Load the audio with original sampling rate
            if sr != 48000:
                audio_sample = librosa.resample(audio_sample, orig_sr=sr, target_sr=48000)

            inputs = self.processor(audios=audio_sample, return_tensors="pt", sampling_rate=48000).to(self.device)

            audio_embedding = self.model.get_audio_features(**inputs)
            audio_embedding = audio_embedding.detach().cpu()
            torch.save(audio_embedding, feat_save_path)

        return audio_embedding


    @torch.no_grad()
    def get_text_embedding(self, text, lang, use_clip=False):
        SPACER = "@@@"
        text = text.split(SPACER)[0]
        text_id = self.text_to_id[text]
        if use_clip:
            feat_save_path = os.path.join(self.feature_save_dir, self.data_split, lang, self.audio_model, f'{text_id:05d}.pt')
        else:
            feat_save_path = os.path.join(self.feature_save_dir, self.data_split, lang, self.text_model, f'{text_id:05d}.pt')

        os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)

        if os.path.exists(feat_save_path):
            text_embedding = torch.load(feat_save_path)
        elif use_clip:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_embedding = self.model.get_text_features(**inputs)
            text_embedding = text_embedding.detach().cpu()
            torch.save(text_embedding, feat_save_path)
        else:
            text_embedding = self.labse_model.encode([text])
            text_embedding = torch.from_numpy(text_embedding)
            torch.save(text_embedding, feat_save_path)

        return text_embedding

