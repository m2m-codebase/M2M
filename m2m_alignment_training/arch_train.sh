
WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse_skip_conn_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method skip_conn --loss_method mse --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse_klclip_skip_conn_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method skip_conn --loss_method mse_klclip --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse_klclip_L4_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --num_layers 4 --loss_method mse_klclip --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_250_mse_klclip_L1_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 1 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_250_eng_cosine_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method eng_cosine --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_250_l1_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method l1 --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

