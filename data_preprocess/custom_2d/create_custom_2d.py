import argparse
import json
import os
import sys
import io
from multiprocessing import Pool
from pathlib import Path
import shutil
import cairosvg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import vtracer
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch
from PIL import Image
from shapely.geometry import Polygon
from skimage import measure
from tqdm import tqdm
import albumentations as A
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_utils import resort_corners
from stru3d.stru3d_utils import type2id

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2

cv2.setNumThreads(0)
instance_id = 0

import torch

torch.set_num_threads(1)

IMAGE_AUGMENTATION_SCALE_FACTOR = 1.5


CUBICASA_MAPPER = {
    # 1. LivingRoom
    "living": 1, "living_dining": 1, "dining_living": 1, "lounge": 1, "relax": 1, "fe_lr": 1, "fe_lr_hl": 1,
    # 2. Bedroom
    "bedroom_a": 2, "bedroom_b": 2, "master_bedroom": 2, "bd": 2, "bd_a": 2, "bd_b": 2, "bd_c": 2, "loft": 2, "attic": 2, "den": 2, "study": 2,
    # 3. Kitchen
    "kitchen": 3, "kitchen_dining": 3, "dining": 3, "dinning_kitchen_living_room": 3, "cook": 3, "live_cook_entry": 3, "dr_kt_lr_fe": 3, "kitchen_entry_wd": 3, "live_dine_cook_entry": 3, "live_dine_cook_entry_bd": 3, "entry_family_dining_kitchen": 3, "entry_dining_living_room_kitchen": 3, "entry_living_room_kitchen": 3, "fe_dn_lr_kt": 3,
    # 4. Bathroom
    "bathroom_a": 4, "bathroom_b": 4, "master_bathroom": 4, "ba": 4, "ba_a": 4, "ba_b": 4, "ba_c": 4, "ensuite": 4, "ens": 4,
    # 5. Corridor
    "hall": 5, "desk_hall": 5, "live_dine": 5, "live_dine_entry": 5,
    # 6. Balcony
    "balcony": 6, "bl": 6, "patio_balcony": 6, "window_seat": 6,
    # 7. Elevator
    # 8. Stairs
    "stair": 8, "stair_entry": 8, "entry_stair": 8,
    # 9. UtilityRoom
    "garage": 9, "garage_entry": 9, "laundry": 9, "utility": 9, "mechanical": 9, "engineering_room": 9, "mech": 9, "wh": 9, "wd": 9, "wd_wh": 9, "washer_dryer": 9, "ac": 9, "mud": 9, "coat_mud": 9,
    # 10. Closet
    "closet": 10, "closets": 10, "closet_a": 10, "closet_b": 10, "walk_in_closet": 10, "walk_in_closet_a": 10, "walk_in_closet_b": 10, "master_closet": 10, "walk_in_robe_b": 10, "wic": 10, "storage": 10, "store": 10, "pantry": 10, "pan": 10, "pty": 10, "linen": 10, "lin": 10, "lin_a": 10, "lin_b": 10, "coat": 10, "cl": 10, "cl_a": 10, "cl_b": 10, "cl_c": 10, "cl_d": 10, "cl_e": 10, "ref": 10, "ref_closet": 10,
    # 11. Courtyard
    # 12. Entry
    "entry": 12, "entry_living_room_dining": 12,
}

DEFAULT_CATEGORY = 2

CUBICASA_CATEGORIES = [
    {"id": 1, "name": "LivingRoom", "supercategory": "room"},
    {"id": 2, "name": "Bedroom", "supercategory": "room"},
    {"id": 3, "name": "Kitchen", "supercategory": "room"},
    {"id": 4, "name": "Bathroom", "supercategory": "room"},
    {"id": 5, "name": "Corridor", "supercategory": "room"},
    {"id": 6, "name": "Balcony", "supercategory": "room"},
    {"id": 7, "name": "Elevator", "supercategory": "room"},
    {"id": 8, "name": "Stairs", "supercategory": "room"},
    {"id": 9, "name": "UtilityRoom", "supercategory": "room"},
    {"id": 10, "name": "Closet", "supercategory": "room"},
    {"id": 11, "name": "Courtyard", "supercategory": "room"},
    {"id": 12, "name": "Entry", "supercategory": "room"},
]

CUSTOM_TO_CUBICASA_ID_MAPPER = {
    0: 1,   # LivingRoom -> Living Room
    1: 2,   # Bedroom -> Bedroom
    2: 3,   # Kitchen -> Kitchen
    3: 4,   # Bathroom -> Bath (CubiCasa ID)
    4: 5,   # Corridor -> Hallway
    5: 0,   # Balcony -> Outdoor (or drop if using drop_label_ids)
    6: 10,  # Elevator -> Other Rooms
    7: 10,  # Stairs -> Other Rooms
    8: 7,   # UtilityRoom -> Storage
    9: 7,   # Closet -> Storage
    10: 0,  # Courtyard -> Outdoor
    11: 5,  # Entry -> Hallway
}

CC5K_MAPPING_2 = {
    0: None,
    1: 0,  # Outdoor
    2: None,  # Wall
    3: 1,  # Kitchen
    4: 2,  # Living Room
    5: 3,  # Bed Room
    6: 4,  # Bath
    7: 5,  # Entry
    8: None,  # Railing -> Wall
    9: 6,  # Storage
    10: 7,  # Garage
    11: 8,  # Undefined
    12: 9,  # Window
    13: 10,  # Door
}

CC5K_CLASS_MAPPING_2 = {
    "Outdoor": 0,
    "Kitchen": 1,
    "Living Room": 2,
    "Bed Room": 3,
    "Bath": 4,
    "Entry": 5,
    "Storage": 6,
    "Garage": 7,
    "Undefined": 8,
    "Window": 9,
    "Door": 10,
}


def visualize_room_polygons(mask, room_polygons, class_names, save_path="cubicasa_debug.png", bg_polygons=None):
    """
    Visualize the extracted room polygons.

    Args:
        mask: Original segmentation mask
        room_polygons: Dictionary of room polygons as returned by extract_room_polygons
        figsize: Figure size for the plot
    """
    # Set figure size to exactly 256x256 pixels
    dpi = 100  # Standard screen DPI
    figsize = (mask.shape[1] / dpi, mask.shape[0] / dpi)  # Convert pixels to inches

    # Get unique classes from the mask
    unique_classes = np.unique(mask)

    # Create a discrete colormap
    cmap = plt.cm.get_cmap("gist_ncar", 256)  # nipy_spectral
    # cmap = ListedColormap([cmap(x) for x in np.linspace(0, 1, int(20))])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    plt.imshow(mask, cmap=cmap, interpolation="nearest", alpha=0.6, vmin=0, vmax=20)

    # Plot each room polygon
    for polygon, room_cls in room_polygons:
        polygon_array = np.array(polygon).copy()
        # # flip y
        # polygon_array[:, 1] = mask.shape[0] - polygon_array[:, 1] - 1
        ax.plot(polygon_array[:, 0], polygon_array[:, 1], "k-", linewidth=2)

        # Add room ID label at the centroid
        centroid_x = np.mean(polygon_array[:, 0])
        centroid_y = np.mean(polygon_array[:, 1])
        ax.text(
            centroid_x,
            centroid_y,
            str(room_cls),
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    if bg_polygons is not None:
        # Plot each room polygon
        for polygon, room_cls in bg_polygons:
            polygon_array = np.array(polygon).copy()
            # # flip y
            # polygon_array[:, 1] = mask.shape[0] - polygon_array[:, 1] - 1
            ax.plot(polygon_array[:, 0], polygon_array[:, 1], "c-", linewidth=2)

    # Create custom legend elements
    legend_elements = []
    norm = np.linspace(0, 1, 21)  # int(max(unique_classes))+1

    for i, cls in enumerate(sorted(unique_classes)):
        # if int(cls) == 0:
        #     continue
        # Get the exact same color that imshow uses
        color = cmap(norm[int(cls)])
        # color = cmap(int(cls))

        cls_name = f"{int(cls)}_{class_names[int(cls)]}"
        # You can replace f"Class {cls}" with your actual class names if available
        legend_elements.append(Patch(facecolor=color, edgecolor="black", label=f"{cls_name}", alpha=0.6))

    # Add the legend to the plot
    ax.legend(
        handles=legend_elements,
        loc="best",
        title="Classes",
        fontsize=20,
        markerscale=4,
        title_fontsize=28,
    )

    # plt.title('Room Polygons Extracted from Segmentation Mask')
    plt.axis("equal")
    plt.axis("off")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def config():
    a = argparse.ArgumentParser(description="Generate COCO format data for custom 2d dataset")
    a.add_argument(
        "--data_root", default="2d", type=str, help="path to raw 2d dataset folder"
    )
    a.add_argument("--output", default="coco_custom2d", type=str, help="path to output folder")
    a.add_argument("--disable_augmentation", action="store_true")

    args = a.parse_args()
    return args


def save_image(image_path: Path, output_path: Path, output_val: Path, output_test: Path, mask=None):
    """
    ref: https://github.com/ultralytics/ultralytics/issues/339
    """
    if image_path.name.endswith(".svg"):
        png_data = cairosvg.svg2png(url=image_path._str)
        img = Image.open(io.BytesIO(png_data)).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    img.info.pop("icc_profile", None)

    if mask is not None:
        img_array = np.array(img)
        if len(mask.shape) == 2 and len(img_array.shape) == 3:
            mask = mask[:, :, np.newaxis]
        masked_img = np.where(mask == 0, 255, img_array)
        img = Image.fromarray(masked_img.astype(np.uint8))

    img.save(output_path)
    img.save(output_val)
    img.save(output_test)


def enumerate_plan(path: Path) -> tuple[dict] | None:
    areas_path = path.joinpath("areas")
    images_path = path.joinpath("images")

    annotations =  []
    images = []
    for subpath in images_path.iterdir():
        images.append(subpath)

    
    for area_file in areas_path.iterdir():
        for image in images:
            with open(area_file, "r") as f:
                data = json.load(f)
            
                if image.name == data["image_name"]:
                    annotations.append(
                        {
                            "image": image.absolute(),
                            "area_file": area_file.absolute()
                        }
                    )
    return annotations


def create_coco_bounding_box(bb_x, bb_y, image_width, image_height, bound_pad=2):
    bb_x = np.unique(bb_x)
    bb_y = np.unique(bb_y)
    bb_x_min = np.maximum(np.min(bb_x) - bound_pad, 0)
    bb_y_min = np.maximum(np.min(bb_y) - bound_pad, 0)

    bb_x_max = np.minimum(np.max(bb_x) + bound_pad, image_width - 1)
    bb_y_max = np.minimum(np.max(bb_y) + bound_pad, image_height - 1)

    bb_width = bb_x_max - bb_x_min
    bb_height = bb_y_max - bb_y_min

    coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]
    return coco_bb


def prepare_dict(categories_dict):
    save_dict = {"images": [], "annotations": [], "categories": []}
    for key, value in categories_dict.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        save_dict["categories"].append(type_dict)
    return save_dict


def _prepare_dataset(dataset, coco_json_path: Path, img_folder: Path, start_scene_id: int):
    save_dict = prepare_dict(CC5K_CLASS_MAPPING_2)
    scene_id = 0
    global instance_id

    for item in dataset:
        image_path = item["image"]
        area_path = item["area_file"]

        with open(area_path, "r") as f:
            data = json.load(f)

            img_id = int(scene_id) + start_scene_id
            print("Processing image with id: {}".format(img_id))
            
            room_polygons = []
            for item in data.get("areas", []):
                raw_id = item["id"]
                area_class = item["label_id"]
                class_raw_label = item["raw_label"]
                if class_raw_label == "All" or class_raw_label.lower() == "all":
                    continue
                class_id = CUSTOM_TO_CUBICASA_ID_MAPPER.get(CUBICASA_MAPPER.get(area_class.lower().strip(), DEFAULT_CATEGORY), DEFAULT_CATEGORY)

                polygon_coords = item["polygon_px"]
                room_polygons.append([polygon_coords, class_id])
        
        
            if image_path.name.endswith(".svg"):
                png_data = cairosvg.svg2png(url=str(image_path))
                img = Image.open(io.BytesIO(png_data)).convert("RGB")
            else:
                img = Image.open(image_path).convert("RGB")
            
            new_w, new_h = img.size

            img_array = np.array(img)
            output_image = img_array.copy()
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            inter_image = img_array.copy()
            mask = np.zeros((new_h + 2, new_w + 2), np.uint8)
            cv2.floodFill(inter_image, mask, (0, 0), (255, 255, 255))

            img = Image.fromarray(inter_image.astype(np.uint8))
            img.save(f"{img_folder}/{str(img_id).zfill(5) + '.png'}")
            for poly_ind, (polygon, class_id) in enumerate(room_polygons):  
                poly_shapely = Polygon(polygon)
                area = poly_shapely.area              
                polygon_array = np.array(polygon)
                poly_sorted = resort_corners(polygon_array)

                poly_type = CC5K_MAPPING_2[class_id]
                if poly_type is None:
                    poly_type = DEFAULT_CATEGORY
                rectangle_shapely = poly_shapely.envelope
                polygon = np.array(polygon)

                coco_seg_poly = []
                poly_sorted = resort_corners(polygon)

                for p in poly_sorted:
                    coco_seg_poly += list(p)

                bb_x, bb_y = rectangle_shapely.exterior.xy
                coco_bb = create_coco_bounding_box(bb_x, bb_y, new_w, new_h, bound_pad=2)

                coco_annotation_dict = {
                    "segmentation": [coco_seg_poly],
                    "area": int(round(area, 2)),
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": coco_bb,
                    "category_id": poly_type,
                    "id": instance_id,
                }
                
                save_dict["annotations"].append(coco_annotation_dict)
                instance_id += 1
            save_dict["images"].append({
                "file_name": str(img_id).zfill(5) + ".png",
                "id": img_id,
                "width": new_w,
                "height": new_h,
            })
            scene_id += 1
    
    with open(coco_json_path, "w") as f:
        json.dump(save_dict, f, default=int)
    print(f"COCO-converted dict saved to: {coco_json_path}")


if __name__ == "__main__":
    args = config()

    #
    # prepare directories for train, val and testing
    #
    out_folder = args.output
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    #
    # start dataset processing
    #
    start_scene_id = 3500  # following index of s3d data

    full_plan = []

    for path in Path(args.data_root).iterdir():
        if path.is_dir():
            annotations = enumerate_plan(path)
            full_plan.extend(annotations)
        else:
            print(f"############# {path.name}")

    if args.disable_augmentation == False:
        print(f"Augmentation enabled - performing scaling with factor {IMAGE_AUGMENTATION_SCALE_FACTOR}")
        augmented_annotations = []
        augmented_images_names = []
        for item in full_plan:
            image_path = item["image"]
            if Path(image_path).name in augmented_images_names:
                continue

            if image_path.name.endswith(".svg"):
                png_data = cairosvg.svg2png(url=str(image_path))
                img = Image.open(io.BytesIO(png_data)).convert("RGB")
            else:
                img = Image.open(image_path).convert("RGB")

            img_array = np.array(img)
            image_cv = img_array.copy()
            w, h = img.size

            area_path = item["area_file"]
            final_augmented_polygons = []

            with open(area_path, "r") as f:
                data = json.load(f)

                out_json_path = Path(area_path)
                out_image_path = Path(image_path)
                out_aug_annotation = out_json_path.parent.joinpath(f"aug_{out_json_path.name}")
                out_aug_image = out_image_path.parent.joinpath(f"aug_{out_image_path.name}")
                out_contour_image = out_image_path.parent.joinpath(f"aug_cntrs_{out_image_path.name}.png")
                
                areas = data.get("areas", [])
                keypoints = []
                polygon_raw_ids = []

                for i, item in enumerate(areas):
                    raw_id = item["id"]
                    poly = item["polygon_px"]
                    out_poly = []
                    for point in poly:
                        out_poly.append([point[0], point[1]])  # [x, y]
                    keypoints.append(out_poly)
                    polygon_raw_ids.append(tuple([i, raw_id])) # 0: bd_1 etc

                scale_w = IMAGE_AUGMENTATION_SCALE_FACTOR
                scale_h = IMAGE_AUGMENTATION_SCALE_FACTOR
                new_w = int(w * scale_w)
                new_h = int(h * scale_h)

                scaled_keypoints = []
                for poly in keypoints:
                    new_polygon = []
                    for p in poly:
                        print(p)
                        new_polygon.append(tuple(
                            (int(p[0] * scale_w), int(p[1] * scale_h)) 
                        ))
                    scaled_keypoints.append(new_polygon)
                aug_image = cv2.resize(img_array, (new_w, new_h))

                image_contours = aug_image.copy()
                for index, raw_id in polygon_raw_ids:
                    print(f"{index}: {raw_id}")
                    for i in range(0, len(areas)):
                        if areas[i]["id"] == raw_id:
                            areas[i]["polygon_px"] = scaled_keypoints[index]
                            areas[i]["image_name"] = out_aug_image.name
                            areas[i]["width"] = new_w
                            areas[i]["height"] = new_h
                            random_color = np.random.randint(0, 256, size=3).tolist()
                            original_poly_cv = np.array(areas[i]["polygon_px"], dtype=np.int32).reshape(-1, 1, 2)
                            cv2.drawContours(image_contours, [original_poly_cv], -1, random_color, thickness=3)
                data["areas"] = areas

                cv2.imwrite(out_contour_image, image_contours)
                augmented_annotations.append({
                    "image": out_aug_image.absolute(),
                    "area_file": out_aug_annotation.absolute()
                })
                augmented_images_names.append(out_aug_image.name)

                with open(out_aug_annotation, "w") as f:
                    json.dump(data, f)
                if out_aug_image.name.endswith(".svg"):
                    _, buffer = cv2.imencode('.png', aug_image)
                    image_bytes = buffer.tobytes()
                    svg_string = vtracer.convert_raw_image_to_svg(image_bytes, img_format='png')
                    with open(out_aug_image, "w", encoding="utf-8") as f:
                        f.write(svg_string)
                else:
                    pil_img = Image.fromarray(aug_image)
                    pil_img.save(out_aug_image)
                print(f"Written augmented image into {out_aug_image} with annotation: {out_aug_annotation}")

        full_plan.extend(augmented_annotations)
        print(f"Plan length after adding augmentations: {len(full_plan)} ({len(augmented_annotations)} added)")

    scene_id = 0
    labels = [Path(os.path.join("./", annotation["area_file"])).parent.parent.name for annotation in full_plan]

    #
    # test / train split happens here
    #
    train_dataset, test_dataset = train_test_split(
        full_plan,
        test_size=0.2,
        shuffle=True,
        stratify=labels
    )

    for i in range(0, 1):
        train_fold_items = train_dataset
        test_fold_items = test_dataset

        annotation_out_folder = os.path.join(out_folder, "annotations")
        if not os.path.exists(annotation_out_folder):
            os.mkdir(annotation_out_folder)

        train_path = os.path.join(annotation_out_folder, "train.json")
        val_path = os.path.join(annotation_out_folder, "val.json")
        test_path = os.path.join(annotation_out_folder, "test.json")

        img_train_path = os.path.join(out_folder, "train")
        if not os.path.exists(img_train_path):
            os.mkdir(img_train_path)
        
        img_test_path = os.path.join(out_folder, "test")
        if not os.path.exists(img_test_path):
            os.mkdir(img_test_path)

        _prepare_dataset(train_fold_items, train_path, img_train_path, start_scene_id=start_scene_id)
        _prepare_dataset(test_fold_items, test_path, img_test_path, start_scene_id=start_scene_id+len(train_fold_items))
        shutil.copy(test_path, val_path) # val file should still be present in final dataset, even though validation is performed after training separately
        print(f"Converted dataset written to: {out_folder}")
    