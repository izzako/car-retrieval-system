
from transformers import AutoModelForObjectDetection, AutoImageProcessor, pipeline
from tqdm import tqdm
import os
import json
from PIL import Image
import glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

image_folder = "./data/frame_dir"
json_output_folder = "./output/frame_json"
batch_size = 8  # adjust as needed

# 🧪 Create output folder
os.makedirs(json_output_folder, exist_ok=True)

MODEL_NAME = "izzako/detr-resnet-50-finetuned-IDD_Detection"

# Load processor and model manually
print("Load and build model.... {}")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",          # Let HF decide best dtype
    device_map="cuda"            # Avoid meta tensor issues
)

# Build the pipeline
obj_detector = pipeline(
    "object-detection",
    model=model,
    image_processor=processor,
    # device=0  # or -1 for CPU
)

# 📂 Load image paths
image_paths = sorted(list(Path(image_folder).glob("*.jpg")))


# 🔁 Batch inference and save to JSON
print("Start Inference...")
b=0
for i in tqdm(range(0, len(image_paths), batch_size),):
    b+=1
    batch_paths = image_paths[i:i+batch_size]
    batch = [Image.open(p).convert("RGB") for p in batch_paths]
    results = obj_detector(batch)

    for path, prediction in zip(batch_paths, results):
        filename = Path(path).stem + ".json"
        output_path = Path(json_output_folder) / filename

        # 📝 Save prediction to JSON
        with open(output_path, "w") as f:
            json.dump(prediction, f, indent=2)

    print(f"✅ Saved: batch {b}/{round(len(image_paths)/batch_size)}")