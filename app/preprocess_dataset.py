import os
import cv2
import shutil
from tqdm import tqdm

DATASET_PATH = r"D:\Major Project\RDataSet"

def is_image_corrupt(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return True
        return False
    except:
        return True

def fix_image_dimensions(img, min_side=416):
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        img = cv2.resize(img, (640, 640))
    return img

def ensure_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def clean_split(split_path):
    images_path = os.path.join(split_path, "images")
    labels_path = os.path.join(split_path, "labels")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    corrupt_count = 0
    missing_label_count = 0

    for file in tqdm(os.listdir(images_path), desc=f"Processing {split_path}"):
        img_path = os.path.join(images_path, file)
        label_path = os.path.join(labels_path, file.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Delete corrupt images
        if is_image_corrupt(img_path):
            os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            corrupt_count += 1
            continue

        # Load and enhance
        img = cv2.imread(img_path)
        img = ensure_rgb(img)
        img = fix_image_dimensions(img)

        # Overwrite cleaned image
        cv2.imwrite(img_path, img)

        # Ensure label file exists
        if not os.path.exists(label_path):
            missing_label_count += 1
            with open(label_path, "w") as f:
                pass  # create empty label (later can be removed)

    return corrupt_count, missing_label_count


def main():
    splits = ["train", "valid", "test"]
    total_corrupt = 0
    total_missing = 0

    for split in splits:
        split_path = os.path.join(DATASET_PATH, split)
        corrupt, missing = clean_split(split_path)
        total_corrupt += corrupt
        total_missing += missing

    print("\nSUMMARY:")
    print(f"Corrupt images removed: {total_corrupt}")
    print(f"Missing label files created: {total_missing}")

if __name__ == "__main__":
    main()
