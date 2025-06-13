import shutil
import random
import os

def split_dataset(source_dir, target_dir, split_ratio=0.8):
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = os.listdir(cls_path)
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for phase, img_list in zip(['train', 'val'], [train_imgs, val_imgs]):
            dest_dir = os.path.join(target_dir, phase, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(cls_path, img), os.path.join(dest_dir, img))

# 자동 분할 실행
split_dataset("./all", "./data", split_ratio=0.8)
