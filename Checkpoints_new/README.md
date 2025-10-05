# Checkpoint Information  
All checkpoints only contain linear layers please load in conjunction with actual encoders listed below (training code allows to do inference and supports loading these checkpoints)

---

## Image-Text Retrieval Checkpoints 

| Text Encoder    | Image Encoder       | Checkpoint Name                        |
|-----------------|---------------------|----------------------------------------|
| LaBSE           | Jina-CLIP-v1         | 250K_mse_klclip_jina                    |
| Jina-Text-v3    | Jina-CLIP-v1         | en_250K_mse_klclip_jina_jinaTextv3       |
| M-MiniLM        | Jina-CLIP-v1         | en_250K_mse_klclip_jina_multiMiniLM      |
| M-MPNET         | Jina-CLIP-v1         | en_250K_mse_klclip_jina_multiMpnet       |
| M-MPNET         | OpenAI-CLIP          | en_250K_mse_klclip_cliplarge_multiMpnet  |
| M-MPNET         | KakaoBrain-ALIGN     | en_250K_mse_klclip_align_multiMpnet      |  


---

## Audio-Text Retrieval Checkpoints

| Text Encoder | Audio Encoder       | Checkpoint Name                                      |
|--------------|---------------------|-----------------------------------------------------|
| M-MPNET      | [CLAP-HTSAT-FUSED](https://huggingface.co/laion/clap-htsat-fused)     | clap_ht_fused_multiMpnet_audiocaps_clotho_wavcaps    |
| M-MPNET      | [CLAP-General](https://huggingface.co/laion/larger_clap_general)         | clap_general_multiMpnet_audiocaps_clotho_wavcaps     |  



---  

## Cross-lingual Image Generation Checkpoints  

| Text Encoder | Image Generation Model | Checkpoint Name                         |
|--------------|------------------------|------------------------------------------|
| M-MPNET      | FLUX.1-dev              | flux_en_250K_mse_clip_multiMpnet          |  

---

## Architecture Ablation:  

| Text Encoder | Image Encoder | Checkpoint Name                      | Loss                   | MLP Layers | Skip Connection |
|--------------|---------------|--------------------------------------|------------------------|------------|-----------------|
| M-MPNET   | Jina-CLIP-v1     | 250K_mse_jina_multiMpnet              | MSE                    | 2          | No              |
| M-MPNET   | Jina-CLIP-v1     | 250K_mse_skip_conn_jina_multiMpnet    | MSE                    | 2          | Yes             |
| M-MPNET   | Jina-CLIP-v1     | en_250K_mse_klclip_jina_multiMpnet    | 44*MSE + 1*L_str        | 2          | No              |
| M-MPNET   | Jina-CLIP-v1     | 250K_mse_klclip_skip_conn_jina_multiMpnet | 44*MSE + 1*L_str     | 2          | Yes             |
| M-MPNET   | Jina-CLIP-v1     | 250K_mse_klclip_L4_jina_multiMpnet    | 44*MSE + 1*L_str        | 4          | No              |
| M-MPNET   | Jina-CLIP-v1     | en_250_mse_klclip_L1_jina_multiMpnet  | 44*MSE + 1*L_str        | 1          | No              |
| M-MPNET   | Jina-CLIP-v1     | en_250_eng_cosine_jina_multiMpnet     | Similarity Loss         | 2          | No              |
| M-MPNET   | Jina-CLIP-v1     | en_250_l1_jina_multiMpnet             | L1                     | 2          | No              |
| M-MPNET   | Jina-CLIP-v1     | en_250K_1mse_1klclip_jina_multiMpnet  | 1*MSE + 1*L_str         | 2          | No              |  

---

## Scaling Experiments

| Text Encoder | Image Encoder   | Checkpoint Name                        | No. of Sentences |
|--------------|-----------------|----------------------------------------|------------------|
| M-MPNET       | Jina-CLIP-v1    | en_1K_mse_klclip_jina_multiMpnet        | 1K               |
| M-MPNET       | Jina-CLIP-v1    | en_5K_mse_klclip_jina_multiMpnet        | 5K               |
| M-MPNET       | Jina-CLIP-v1    | en_10K_mse_klclip_jina_multiMpnet       | 10K              |
| M-MPNET       | Jina-CLIP-v1    | en_50K_mse_klclip_jina_multiMpnet       | 50K              |
| M-MPNET       | Jina-CLIP-v1    | en_100K_mse_klclip_jina_multiMpnet      | 100K             |
| M-MPNET       | Jina-CLIP-v1    | en_250K_mse_klclip_jina_multiMpnet      | 250K             |
| M-MPNET       | Jina-CLIP-v1    | en_2M_mse_klclip_jina_multiMpnet        | 2M               |  

