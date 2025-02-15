WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse_klclip_jina --image_model jinav1 --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_250K_mse_klclip_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_250K_mse_klclip_jina_jinaTextv3 --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model jinaTextv3 --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_250K_mse_klclip_jina_multiMiniLM --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMiniLM --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_250K_mse_klclip_cliplarge_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model clip --text_model multiMpnet --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_250K_mse_klclip_align_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model align --text_model multiMpnet --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

