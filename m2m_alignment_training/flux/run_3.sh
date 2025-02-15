## Persian/farsi
CUDA_VISIBLE_DEVICES=1 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "5" --batch_size 48


## Hebrew
CUDA_VISIBLE_DEVICES=1 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "2" --batch_size 48

## Indonesian
CUDA_VISIBLE_DEVICES=1 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "3" --batch_size 48 

#CUDA_VISIBLE_DEVICES=3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col caption --batch_size 48

## French
#CUDA_VISIBLE_DEVICES=3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "0" --batch_size 48
