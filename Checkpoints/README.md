# Checkpoint Information  
All checkpoints only contain linear layers please load in conjunction with actual encoders listed below (training code allows to do inference and supports loading these checkpoints)

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

