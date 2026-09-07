import argparse
import json
import os
import sys
import io
from multiprocessing import Pool
from pathlib import Path
import cairosvg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch
from PIL import Image
from shapely.geometry import Polygon
from skimage import measure
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_utils import resort_corners
from stru3d.stru3d_utils import type2id

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2

cv2.setNumThreads(0)

import torch

torch.set_num_threads(1)


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

DEFAULT_CATEGORY = 9 

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


def fill_holes_in_mask(binary_mask):
    """
    Fill 0-pixels in a binary mask that are completely surrounded by 1-pixels.

    Args:
        binary_mask (numpy.ndarray): Binary mask with 0 and 1 values.

    Returns:
        numpy.ndarray: Binary mask with holes filled.
    """
    # Ensure the mask is binary (0 and 1)
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill the contours
    filled_mask = binary_mask.copy()
    cv2.fillPoly(filled_mask, contours, 1)

    return filled_mask


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Ref: https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def extract_icon_cv2(mask, start_cls_id=11, skip_classes=[]):
    room_ids = np.unique(mask)
    room_polygons = []
    new_mask = np.zeros(mask.shape)

    # window, door
    for room_id in room_ids:
        if room_id in skip_classes:
            continue
        true_room_id = int(room_id) + start_cls_id
        # Create binary mask for this room
        room_mask = (mask == room_id).astype(np.uint8)
        new_mask = np.where(room_mask, true_room_id, 0)

        # Find contours using OpenCV
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # # Get the largest contour
            # largest_contour = max(contours, key=cv2.contourArea)
            for cnt in contours:
                polygon = [tuple(point[0]) for point in cnt]
                if len(polygon) < 3:
                    continue

                poly = Polygon(polygon)
                simplified_poly = poly.simplify(tolerance=0.5, preserve_topology=True)
                simplified_poly = list(simplified_poly.exterior.coords)
                room_polygons.append([simplified_poly, true_room_id])

    return room_polygons, new_mask


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
    a.add_argument("--disable_wd2line", action="store_true")

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


def remove_polygons_by_type(polygons, skip_types=[]):
    new_room_polygons = []
    for polygon, poly_type in polygons:
        if poly_type in skip_types:
            continue
        new_room_polygons.append([polygon, poly_type])
    return new_room_polygons


def merge_rooms_and_icons(room_polygons, icon_polygons):
    new_icon_polygons = []
    for poly, poly_type in icon_polygons:
        new_icon_polygons.append([poly, poly_type + 11])

    return room_polygons + new_icon_polygons


def enumerate_plan(path: Path) -> tuple[dict] | None:
    areas_path = path.joinpath("areas")
    images_path = path.joinpath("images")

    annotations =  []
    images = []
    for subpath in images_path.iterdir():
        images.append(subpath)

    for image in images:
        for area_file in areas_path.iterdir():
            if area_file.stem.startswith(image.stem) and area_file.name.endswith(".json"):
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
    temp_categories = []
    for item in categories_dict:
        id = item["id"]
        name = item["name"]
        temp_categories.append({
            "supercategory": "room", 
            "id": int(id) - 1, 
            "name": str(name)
        })
    temp_categories.sort(key=lambda x: x["id"])
    save_dict["categories"] = temp_categories
    return save_dict


def prepare_dict2(categories_dict):
    save_dict = {"images": [], "annotations": [], "categories": []}
    for key, value in categories_dict.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        save_dict["categories"].append(type_dict)
    return save_dict


def _prepare_dataset(dataset, image_size: int, coco_json_path: Path, img_folder: Path, start_scene_id: int):
    save_dict = prepare_dict(CUBICASA_CATEGORIES)
    scene_id = 0
    instance_id = 0

    for item in dataset:
        image_path = item["image"]
        area_path = item["area_file"]

        with open(area_path, "r") as f:
            data = json.load(f)
            #print("processing image {} with annotation {} ...".format(image_path, area_path))

            img_id = int(scene_id) + start_scene_id
            print("Processing image with id: {}".format(img_id))
            
            room_polygons = []
            for item in data.get("areas", []):
                raw_id = item["id"]
                area_class = item["label_id"]
                class_id = CUBICASA_MAPPER.get(area_class.lower().strip(), DEFAULT_CATEGORY)
                print("Mapped {} to {}".format(area_class.lower().strip(), class_id))
                assert class_id < 13
                
                polygon_coords = item["polygon_px"]
                room_polygons.append([polygon_coords, class_id])
        
        
            if image_path.name.endswith(".svg"):
                png_data = cairosvg.svg2png(url=str(image_path))
                img = Image.open(io.BytesIO(png_data)).convert("RGB")
            else:
                img = Image.open(image_path).convert("RGB")
            
            old_w = data["width"]
            old_h = data["height"]
            #print("resizing to {}x{}".format(image_size, image_size))
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
            new_w, new_h = img.size
            scale_w = image_size / old_w
            scale_h = image_size / old_h
            #print("scale: {}x{}".format(scale_w, scale_h))

            scaled_room_polygons = []
            for poly, class_name in room_polygons:
                scaled_poly = []
                for pt in poly:
                    x, y = pt
                    new_x = int(round(x * scale_w))
                    new_y = int(round(y * scale_h))
                    new_x = max(0, min(new_x, image_size - 1))
                    new_y = max(0, min(new_y, image_size - 1))
                    
                    scaled_poly.append((new_x, new_y))
                scaled_room_polygons.append((scaled_poly, class_name))
            room_polygons = scaled_room_polygons
            
            binary_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            for poly, _ in room_polygons:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(binary_mask, [pts], color=1)

            filled_mask = fill_holes_in_mask(binary_mask)
            fixed_mask_size = 512
            binary_mask_1024 = cv2.resize(
                filled_mask,
                (fixed_mask_size, fixed_mask_size),
                interpolation=cv2.INTER_NEAREST,
            )
            img.info.pop("icc_profile", None)

            if binary_mask_1024 is not None:
                img_array = np.array(img)
                if len(binary_mask.shape) == 2 and len(img_array.shape) == 3:
                    binary_mask_1024 = binary_mask_1024[:, :, np.newaxis]
                img_1024 = cv2.resize(
                    img_array,
                    (fixed_mask_size, fixed_mask_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                masked_img = np.where(binary_mask_1024 == 0, 255, img_1024)
                img = Image.fromarray(masked_img.astype(np.uint8))

            img.save(f"{img_folder}/{str(img_id).zfill(5) + '.png'}")
        
            coco_annotation_dict_list = []
        
            for poly_ind, (polygon, class_id) in enumerate(room_polygons):                
                polygon_array = np.array(polygon)
                poly_sorted = resort_corners(polygon_array)
                
                coco_seg_poly = []
                for p in poly_sorted:
                    coco_seg_poly += [int(p[0]), int(p[1])] # was float before
                

                #if len(coco_seg_poly) < 6:
                #    continue

                pairs = [(coco_seg_poly[i], coco_seg_poly[i+1]) for i in range(0, len(coco_seg_poly), 2)]
                poly_shapely = Polygon(pairs)

                area = poly_shapely.area
                
                #if area < 100: 
                #    continue
                    
                #bb_x, bb_y = poly_shapely.envelope.exterior.xy
                #coco_bb = create_coco_bounding_box(bb_x, bb_y, new_w, new_h, bound_pad=2)
                xs = coco_seg_poly[0::2]
                ys = coco_seg_poly[1::2]
                x_min, y_min = min(xs), min(ys)
                width = max(xs) - x_min
                height = max(ys) - y_min
                coco_bb = [int(x_min), int(y_min), int(width), int(height)]
                
                coco_annotation_dict = {
                    "segmentation": [coco_seg_poly],
                    "area": int(round(area, 2)),
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": coco_bb,
                    "category_id": int(class_id) - 1,
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
        json.dump(save_dict, f)
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

    #print(f"{full_plan}")
    #print(f"Full plan length: {len(full_plan)}")

    scene_id = 0
    instance_id = 0
    image_size = 512

    #
    # test / train split happens here
    #
    #train_dataset, test_dataset = train_test_split(
    #    full_plan,
    #    test_size=0.2,
    #    shuffle=True
    #)

    k_fold = KFold(n_splits=5, shuffle=True)
    for fold_idx, (train_index, test_index) in enumerate(k_fold.split(full_plan)):
        print(f"--- Fold {fold_idx + 1} ---")
        print(f"Test items count: {len(test_index)}")
        print(f"Train items count: {len(train_index)}")

        #train_fold_items = full_plan[train_index]
        #test_fold_items = full_plan[test_index]

        train_fold_items = [full_plan[i] for i in train_index]
        test_fold_items = [full_plan[i] for i in test_index]

        fold_path = os.path.join(out_folder, f"fold_{fold_idx}")
        if not os.path.exists(fold_path):
            os.mkdir(fold_path)

        annotation_out_folder = os.path.join(fold_path, "annotations")
        if not os.path.exists(annotation_out_folder):
            os.mkdir(annotation_out_folder)

        train_path = os.path.join(annotation_out_folder, "train.json")
        test_path = os.path.join(annotation_out_folder, "test.json")

        img_train_path = os.path.join(annotation_out_folder, "train")
        if not os.path.exists(img_train_path):
            os.mkdir(img_train_path)
        
        img_test_path = os.path.join(annotation_out_folder, "test")
        if not os.path.exists(img_test_path):
            os.mkdir(img_test_path)

        _prepare_dataset(train_fold_items, image_size, train_path, img_train_path, start_scene_id=start_scene_id)
        _prepare_dataset(test_fold_items, image_size, test_path, img_test_path, start_scene_id=start_scene_id+len(train_index))
        print(f"Prepared fold: {fold_path}")
    