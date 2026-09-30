from detectron2.data.datasets import load_coco_json
import json
import os

dataset_path = os.path.abspath("~/Raster2Seq/data_preprocess/custom_2d/output_stratified_16/")
json_path = os.path.join(dataset_path, "annotations", "train.json")
train_img_folder = os.path.join(dataset_path, "train")

# perform sanity check using detectron2
try:
    dicts = load_coco_json(json_path, train_img_folder, dataset_name="debug_dataset")
    print(f"--- DETECTRON2 LOADING RESULT ---")
    print(f"Successfully loaded: {len(dicts)}")
    valid_anns = sum(len(d["annotations"]) for d in dicts)
    print(f"Valid annotations: {valid_anns}")
    
    if len(dicts) > 0 and valid_anns == 0:
        print("No annotations")
        with open(json_path, "r") as f:
            raw_data = json.load(f)
            print("\n--- First annotation from JSON ---")
            print(json.dumps(raw_data["annotations"][0], indent=2))
            print("\n--- Categories from JSON ---")
            print(json.dumps(raw_data.get("categories", []), indent=2))
except Exception as e:
    print(f"Detectron2 parsing error: {e}")
