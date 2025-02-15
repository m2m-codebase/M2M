import glob
import pandas as pd
import json
import os

img_path = "val2014"
annotation_path = "annotations/captions_val2014.json"


all_images = glob.glob(f"{img_path}/*.jpg")
len(all_images)

with open(annotation_path, "r") as f:
    val_captions = json.load(f)
len(val_captions["images"])


print(val_captions["annotations"][0])
print(val_captions["images"][0])


data_dict = {}

# Create a mapping of image IDs to captions
image_id_to_caption = {annotation["image_id"]: annotation["caption"] for annotation in val_captions["annotations"]}

# Build data_dict using the mapping
for image in val_captions["images"]:
    image_id = image["id"]
    if image_id in image_id_to_caption:
        if os.path.exists(os.path.join("val2014", image["file_name"])):
            data_dict[image["file_name"]] = image_id_to_caption[image_id]

len(data_dict)


df_dict = {"file_name": list(data_dict.keys()), "caption": list(data_dict.values())}

data_df = pd.DataFrame(df_dict)
data_df.head()


N = 30_000
data_df = data_df.sample(N, random_state=2024)
assert len(data_df) == N
data_df.to_csv(f"coco_{N}_randomly_sampled_2014_val.csv", index=False)
print(data_df)


# make image split for FID
save_dir = f"{N}_COCO_Image_split"
os.makedirs(save_dir, exist_ok=True)
for idx, row in data_df.iterrows():
    save_path = os.path.join(save_dir, row['file_name'])
    src_path = os.path.join("val2014", row["file_name"])

    cmd = f"cp {src_path} {save_path}"
    os.system(cmd)

