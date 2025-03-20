import os
import glob
import random
import torch


image_folder = "/data/xueyanz/data/3dgs/train/images"
test_root = "/data/xueyanz/data/3dgs/train/test_names.da"
train_subset_root = "/data/xueyanz/data/3dgs/train/train_subset_names.da"

full_image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
train_image_names = [x.split("/")[-1] for x in full_image_paths]
test_image_names = torch.load(test_root)

train_image_subset = []
for train_image_name in train_image_names:
    if train_image_name not in test_image_names:
        if random.random() < 0.1:
            train_image_subset.append(train_image_name.split(".")[0])
        
torch.save(train_image_subset, train_subset_root)
print(len(train_image_subset))