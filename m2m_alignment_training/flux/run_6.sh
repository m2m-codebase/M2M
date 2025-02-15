#CUDA_VISIBLE_DEVICES=6 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method T5_only --text_col caption --batch_size 48

## Russian
CUDA_VISIBLE_DEVICES=4 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "6" --batch_size 48


## Spanish
CUDA_VISIBLE_DEVICES=4 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "7" --batch_size 48

## Greek
CUDA_VISIBLE_DEVICES=4 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "1" --batch_size 48


## Indonesian
CUDA_VISIBLE_DEVICES=4 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "3" --batch_size 48 --reverse 
