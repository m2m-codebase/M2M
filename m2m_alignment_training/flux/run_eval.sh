#Hindi
#
#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col hin_Deva --image_save_dir ./30K_final_gen/flux_mpnet_hin_Dev_2_512_a_photo_of  
# clip only
#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val.csv --method clip_only --text_col caption --image_save_dir ./30K_final_gen/flux_baseline_clip_only_512_an_image_of --prompt_2 "An image of: " --nhead 25 


#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method clip_only --text_col caption --image_save_dir ./30K_final_gen/flux_baseline_clip_only_512_a_photo_of   

#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val.csv --method clip_only --text_col caption --image_save_dir ./30K_final_gen/flux_baseline_clip_only_512_a_picture_of --prompt_2 "A pictue of: " --nhead 25 

# -------------------------------------------------

# CLIP+T5=Baseline
#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val.csv --method baseline --text_col caption --image_save_dir ./30K_final_gen/flux_baseline_512_an_image_of --prompt_2 "An image of: " --nhead 25


CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method baseline --text_col caption 


#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val.csv --method baseline --text_col caption --image_save_dir ./30K_final_gen/flux_baseline_512_a_picture_of --prompt_2 "A picture of: " --nhead 25

# -------------------------------------------------

# T5 Only
#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val.csv --method T5_only --text_col caption --image_save_dir ./30K_final_gen/flux_baseline_T5_only_512_an_image_of --prompt_2 "An image of: " --nhead 25

CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method T5_only --text_col caption 

#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val.csv --method T5_only --text_col caption --image_save_dir ./30K_final_gen/flux_baseline_T5_only_512_a_picture_of --prompt_2 "A picture of: " --nhead 25

# -----------------------------------

# Labse
#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name flux_en_250K_mse_clipflux_labse --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val_indictrans2.csv --method Labse --text_col hin_Deva --image_save_dir ./30K_final_gen/flux_labse_hin_Deva_512_an_image_of --prompt_2 "An image of: " --nhead 25
#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name flux_en_250K_mse_clipflux_labse --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val_indictrans2.csv --method Labse --text_col hin_Deva --image_save_dir ./30K_final_gen/flux_labse_hin_Deva_512_a_photo_of  --nhead 25
#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name flux_en_250K_mse_clipflux_labse --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val_indictrans2.csv --method Labse --text_col hin_Deva --image_save_dir ./30K_final_gen/flux_labse_hin_Deva_512_a_picture_of --prompt_2 "A picture of: " --nhead 25

# ----------------------------------------

# Mpnet
#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val_indictrans2.csv --method Labse --text_col hin_Deva --image_save_dir ./30K_final_gen/flux_mpnet_hin_Dev_2_512_an_image_of --prompt_2 "An image of: " --nhead 25 

## Hindi
#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col hin_Deva --image_save_dir ./30K_final_gen/flux_mpnet_hin_Dev_2_512_a_photo_of  

#CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../eval_data/coco2014/coco_500_randomly_sampled_2014_val_indictrans2.csv --method Labse --text_col hin_Deva --image_save_dir ./30K_final_gen/flux_mpnet_hin_Dev_2_512_a_picture_of --prompt_2 "A picture of: " --nhead 25 

## English
CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col caption  

## Korean
CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "4" 

## French
CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "0"

## Russian
CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "6" 

## Spanish
CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "7" 

## Persian/farsi
CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "5" 


## Hebrew
CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "2" 

## Greek
CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "1"

## Indonesian
CUDA_VISIBLE_DEVICES=6,3 python load_model.py --wandb_name model_checkpoint --epoch 10 --csv_file ../../t-code2/all_langs_joined.csv --method Labse --text_col "3" 

