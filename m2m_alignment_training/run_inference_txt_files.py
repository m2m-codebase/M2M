import torch
from typing import Union, List, Optional

from diffusers import FluxPipeline
from model import labse_clip
#from model_simple import labse_clip

from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser

import os
from functools import partial 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import string
from accelerate import infer_auto_device_map
import time

def remove_punc(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def clean_text(x):
    x = str(x)
    x = x.lower()
    x = remove_punc(x)
    x = ' '.join(x.split())
    x = x.replace(' ', '_')
    return x



def save_image_grid(image_paths, captions, txt, output_path):
    # Create a figure with a grid layout
    fig, axes = plt.subplots(2,3, figsize=(12, 8))
    fig.suptitle(txt, fontsize=16, fontweight="bold", y=0.95)

    # Adjust spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    # Add images and captions
    for i, ax in enumerate(axes.flat):
        img = mpimg.imread(image_paths[i])

        ax.imshow(img, aspect='auto')
        ax.set_title(captions[i], fontsize=12)
        ax.axis("off")  # Hide axes

    # Show the grid
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Grid saved as {output_path}")


parser = ArgumentParser()
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--emb_method", type=str, default="no_skip_conn", 
    help="used in model for skip conn or sequential transformation")
parser.add_argument("--wandb_name", type=str, help="wandb run name")
parser.add_argument("--prompt_file", type=str, help="wandb run name")
parser.add_argument("--save_dir", type=str, help="dir where checkpoint", 
        default="./content/drive/MyDrive/base-clip-data/models")
args = parser.parse_args()

LABSE_MODEL_NAME = ""
if "labse" in args.wandb_name.lower():
    print("LABSE used")
    LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
else:
    print("MPNET USED")
    LABSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.bfloat16, 
        device_map='balanced',
        #device_map = device_map,
        #use_accelerate=True
    )
#pipe = pipe.to("cuda")
DEVICE='cuda'
DEVICE2='cpu'
#pipe.enable_sequential_cpu_offload()

# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
def load_labse_model():
    model = labse_clip(hdim=768, args=args)

    labse = SentenceTransformer(LABSE_MODEL_NAME)
    if "labse" in LABSE_MODEL_NAME.lower():
        labse[3] = torch.nn.Identity()
    labse = labse.to(DEVICE)

    save_path = os.path.join(args.save_dir, args.wandb_name, f"epoch_{args.epoch}.pth")
    loaded_layers = torch.load(save_path, map_location='cpu')
    model.load_state_dict(loaded_layers, strict=False)

    print(loaded_layers.keys())
    model = model.to(DEVICE).to(dtype=torch.bfloat16)
    #model = model.to(dtype=torch.bfloat16)
    model.eval()
    return model, labse

@torch.no_grad()
def _get_clip_prompt_embeds(
    self,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    custom_model = None,
    custom_labse = None,
    ):
    
    text_emb = torch.from_numpy(custom_labse.encode(prompt)).to(DEVICE).to(dtype=torch.bfloat16)
    #text_emb = torch.from_numpy(custom_labse.encode(prompt)).to(dtype=torch.bfloat16)
    text_emb = custom_model.inference(text_emb)

    #text_emb = text_emb.detach().cpu()

    print('text', text_emb.shape)
    return text_emb
        
# Paths to your images
saved_image_paths = []

#prompt_en = ["A cat holding a sign that says hello world"]
#prompt_en = ["A dog holding a sign that says hello world"]
#prompt_en = ["A cow playing on the grass"]
prompt_en = ["A man with a hat"]#, "An image of: colorful mosaic"]
#prompt_en = ["Playing children"]
#prompt_en = ["Crying children"]
prompt_en = ['This photograph appears to be looking truly wonderful.']
prompt_en = ['Woman with umbrella on a rainy day, near a broken, abandoned umbrella.']
prompt_en = ["The group of friends is enjoyingplaying the video games."]
prompt_en = ["A truck is shown decaying among flowers without a window."]

with open(args.prompt_file, 'r') as f:
    lines = f.readlines()
lines = [l.strip() for l in lines]
prompt_en = lines[:1]
lines  = lines[1:]

dir_name = os.path.join(args.wandb_name, f"epoch_{args.epoch}", clean_text(prompt_en[0]))
save_names_en = [clean_text(x)+'.png' for x in prompt_en]

prompt = [
    #"A cat holding a sign that says hello world",
    #"안녕 세상이라고 적힌 표지판을 들고 있는 고양이",
    #"एक बिल्ली जो हैलो वर्ल्ड का बोर्ड पकड़े हुए है",
    #"ஹலோ வேர்ல்ட் என்று சொல்லும் பலகையை வைத்திருக்கும் பூனை",
    
    #"A dog holding a sign that says hello world",
    #"안녕하세요 세상이라고 적힌 표지판을 들고 있는 개",
    #"एक कुत्ता जो हैलो वर्ल्ड का बोर्ड पकड़े हुए है",
    #"ஹலோ வேர்ல்ட் என்று சொல்லும் பலகையை வைத்திருக்கும் நாய்",
    
    prompt_en[0],
    #"풀밭에서 노는 소",
    #"घास पर खेलती गाय",
    #"புல்லில் விளையாடும் மாடு",

    
    
    ]
prompt  += lines

lang = ['en', 'fr', 'el', 'he', 'id', 'ko', 'fa', 'ru', 'es', 'hi']
#lang = ['hi']
save_names = [
    f"{l}_{x}"
    for x in save_names_en
        for l in lang
]

prompt_2 = ["An image of: "]#, prompt_en[0]]
prompt_2 = ["add a book on bed: "]#, prompt_en[0]]

#prompt_2 = ["A vibrant mosaic photo of: "]#, prompt_en[0]]
#prompt_2 = ["A green-themed photo of: "]#, prompt_en[0]]
#prompt_2 = ["A cartoon photo of: "]#, prompt_en[0]]

#prompt_2 = ["Keepig rest of the objects same, only add Van Gogh style to the image"]#, prompt_en[0]]
#prompt_2 = ["A Picasso-style photo of: "]
#prompt_2 = ["A Van Gogh-style photo of: "]
#prompt_2 = ["A Leonardo da Vinci-style photo of: "]
#prompt_2 = ["A picture of: "]#, prompt_en[0]]

model, labse = load_labse_model()

model = model.to(DEVICE)
labse = labse.to(DEVICE)


#print(model)

#clip_emb = pipe._get_clip_prompt_embeds(prompt_en)
#labse_emb = _get_clip_prompt_embeds(pipe, prompt, custom_model=model, custom_labse=labse)

seed, seed1, seed2 = 0,0,0
#seed, seed1, seed2 = 1,1,1

#print(torch.nn.functional.mse_loss(clip_emb, labse_emb))
base_dir = os.path.join('flux_gen_images', dir_name)
os.makedirs(base_dir, exist_ok=True)
save_images_en = [os.path.join(base_dir, 'CLIP_ONLY_BASELINE_'+x) for x in save_names_en]
saved_image_paths += save_images_en

import gc

height = 512
width = 512
max_seq_len = 512

with torch.no_grad():
    for idx, (save_p, p) in enumerate(zip(save_images_en, prompt_en)):
        image = pipe(
            [p],
            prompt_2=prompt_2,
            #prompt_2=[prompt_2[idx]],
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=10,
            max_sequence_length=max_seq_len,
            generator=torch.Generator("cpu").manual_seed(seed)
        )
        
        image.images[0].save(save_p)
        print(f"CLIP Image saved at: {save_p}")

    save_images_en = [os.path.join(base_dir, 'BASELINE_'+x) for x in save_names_en]
    saved_image_paths += save_images_en

    for save_p, p in zip(save_images_en, prompt_en):
        image = pipe(
            [p],
            prompt_2=[p],
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=10,
            max_sequence_length=max_seq_len,
            generator=torch.Generator("cpu").manual_seed(seed1)
        )
        
        image.images[0].save(save_p)
        print(f"CLIP Image saved at: {save_p}")

    save_images_en = [os.path.join(base_dir, 'T5_'+x) for x in save_names_en]
    saved_image_paths += save_images_en

    for save_p, p in zip(save_images_en, prompt_en):
        image = pipe(
            prompt_2,
            prompt_2=[p],
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=10,
            max_sequence_length=max_seq_len,
            generator=torch.Generator("cpu").manual_seed(seed1)
        )
        
        image.images[0].save(save_p)
        print(f"CLIP Image saved at: {save_p}")
    # LABSE generationn
    pipe._get_clip_prompt_embeds = partial(_get_clip_prompt_embeds, pipe, custom_model=model, custom_labse=labse)

    save_images = [os.path.join(base_dir, x) for x in save_names]
    saved_image_paths += save_images

    for save_p, p in zip(save_images, prompt):
        image = pipe(
            [p],
            prompt_2=prompt_2,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=10,
            max_sequence_length=max_seq_len,
            generator=torch.Generator("cpu").manual_seed(seed2)
        )
        
        image.images[0].save(save_p)
        print(f"Labse Image saved at: {save_p}")

    '''
    cap = [x.split('/')[-1].replace('.png', '').replace(dir_name,'').strip('_') for x in saved_image_paths]
    output_path = os.path.join(base_dir, "all_images.png")
    save_image_grid(saved_image_paths, captions=cap, txt=dir_name, output_path=output_path)
    '''
