import os
import shutil
from tqdm import tqdm

# -------------------------------
# CONFIGURE YOUR PATHS HERE
# -------------------------------
FALL_DATASET = r"D:\Major Project\Adataset"       # class 0
NONFALL_DATASET = r"D:\Major Project\Hdataset"    # class 1
OUTPUT_DATASET = r"D:\Major Project\RDataSet"

# --------------------------------
# INTERNAL FUNCTIONS
# --------------------------------

def ensure_dirs(base):
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(base, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, split, "labels"), exist_ok=True)

def get_label_path(img_path):
    if img_path.endswith(".jpg"):
        return img_path.replace("images", "labels").replace(".jpg", ".txt")
    if img_path.endswith(".png"):
        return img_path.replace("images", "labels").replace(".png", ".txt")
    return None

def remap_label(label_path, new_class_id):
    """Rewrite the class ID for each label file."""
    if not os.path.exists(label_path):
        # Create empty label if missing
        with open(label_path, "w") as f:
            pass
        return

    updated_lines = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                parts[0] = str(new_class_id)  # replace class ID
                updated_lines.append(" ".join(parts))

    with open(label_path, "w") as f:
        f.write("\n".join(updated_lines))

def copy_dataset(src, dst, class_id):
    """Copy train/valid/test folders from src → dst with class remapping."""
    for split in ["train", "valid", "test"]:
        src_img = os.path.join(src, split, "images")
        src_lbl = os.path.join(src, split, "labels")

        dst_img = os.path.join(dst, split, "images")
        dst_lbl = os.path.join(dst, split, "labels")

        for file in tqdm(os.listdir(src_img), desc=f"Merging {split} from {src}"):
            src_img_path = os.path.join(src_img, file)
            new_file = file.replace(" ", "_")  # avoid space issues
            dst_img_path = os.path.join(dst_img, new_file)

            # Copy image
            shutil.copy2(src_img_path, dst_img_path)

            # Handle label
            src_label_path = get_label_path(src_img_path)
            dst_label_path = get_label_path(dst_img_path)

            if src_label_path and os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)

            # Remap label class ID
            remap_label(dst_label_path, class_id)


# ------------------------------
# MAIN PIPELINE
# ------------------------------
print("\n🔧 Setting up output dataset structure...")
ensure_dirs(OUTPUT_DATASET)

print("\n📌 Merging FALL dataset (class 0)...")
copy_dataset(FALL_DATASET, OUTPUT_DATASET, class_id=0)

print("\n📌 Merging NON-FALL dataset (class 1)...")
copy_dataset(NONFALL_DATASET, OUTPUT_DATASET, class_id=1)

print("\n📄 Generating data.yaml...")

yaml_path = os.path.join(OUTPUT_DATASET, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""
train: {OUTPUT_DATASET.replace("\\", "/")}/train/images
val: {OUTPUT_DATASET.replace("\\", "/")}/valid/images
test: {OUTPUT_DATASET.replace("\\", "/")}/test/images

nc: 2
names: ['fall', 'nonfall']
""")

print("\n✅ MERGE COMPLETE!")
print("Your final dataset is ready at:")
print(OUTPUT_DATASET)
print("\nNow you can start YOLO training using:")
print(f'yolo detect train data="{yaml_path}" model=yolov8s.pt epochs=100 imgsz=640 device=0')
