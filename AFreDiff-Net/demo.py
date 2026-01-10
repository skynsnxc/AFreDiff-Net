import os
import shutil
import random

# 原始数据集路径
original_dataset_path = 'CVC-ClinicDB'
original_images_path = os.path.join(original_dataset_path, 'Original')
original_masks_path = os.path.join(original_dataset_path, 'Ground Truth')

# 目标路径
train_images_path = 'CVC-ClinicDB/train/images'
train_masks_path = 'CVC-ClinicDB/train/masks'
val_images_path = 'CVC-ClinicDB/val/images'
val_masks_path = 'CVC-ClinicDB/val/masks'

# 创建目标文件夹
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_masks_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_masks_path, exist_ok=True)

# 获取所有图像文件名
image_files = os.listdir(original_images_path)

# 划分训练集和验证集
random.shuffle(image_files)
train_size = int(len(image_files) * 0.8)
train_files = image_files[:train_size]
val_files = image_files[train_size:]

# 移动训练集文件
for file in train_files:
    image_file_path = os.path.join(original_images_path, file)
    mask_file_path = os.path.join(original_masks_path, file)
    shutil.move(image_file_path, os.path.join(train_images_path, file))
    shutil.move(mask_file_path, os.path.join(train_masks_path, file))

# 移动验证集文件
for file in val_files:
    image_file_path = os.path.join(original_images_path, file)
    mask_file_path = os.path.join(original_masks_path, file)
    shutil.move(image_file_path, os.path.join(val_images_path, file))
    shutil.move(mask_file_path, os.path.join(val_masks_path, file))