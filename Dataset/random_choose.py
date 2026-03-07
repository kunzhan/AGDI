import os
import random
import shutil
from pathlib import Path

def main():

    source_dir = Path("PATH/imagenet/imagenet_val/")  
    save_dir = Path("PATH/random_img/save")
    num_classes = 200                       
    allowed_exts = {".jpg", ".jpeg", ".png"}  


    save_dir.mkdir(parents=True, exist_ok=True)


    class_folders = [f for f in source_dir.iterdir() if f.is_dir()]
    if len(class_folders) < num_classes:
        raise ValueError("Not enough class folders")


    selected_classes = random.sample(class_folders, num_classes)


    for class_folder in selected_classes:

        images = []
        for entry in class_folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in allowed_exts:
                images.append(entry)
        
        if not images:
            print(f"Warning: No images found in {class_folder.name}, skipping")
            continue


        selected_image = random.choice(images)
        

        new_name = f"{class_folder.name}_{selected_image.name}"
        dest_path = save_dir / new_name
        

        shutil.copy(selected_image, dest_path)
        print(f"Copied: {dest_path}")

if __name__ == "__main__":
    main()