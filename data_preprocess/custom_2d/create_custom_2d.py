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

    
    for area_file in areas_path.iterdir():
        for image in images:
            #if image.stem in SKIP_PLANS:
            #    print("Skipping plan: {}".format(image))
            #    continue
            """if area_file.stem.startswith(image.stem) and area_file.name.endswith(".json"):
                annotations.append(
                    {
                        "image": image.absolute(),
                        "area_file": area_file.absolute()
                    }
            )"""
            #prefix = area_file.stem.removesuffix("areas").removesuffix("2d")
            with open(area_file, "r") as f:
                data = json.load(f)
            
                if image.name == data["image_name"]:
                    annotations.append(
                        {
                            "image": image.absolute(),
                            "area_file": area_file.absolute()
                        }
                    )
                
    #if len(annotations) < DATASET_PROVIDER_DEPTH:
    #    return None
    return annotations #[:DATASET_PROVIDER_DEPTH]


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
    save_dict = prepare_dict(CC5K_CLASS_MAPPING_2)
    scene_id = 0
    global instance_id

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
                class_raw_label = item["raw_label"]
                if class_raw_label == "All" or class_raw_label.lower() == "all":
                    print("! ----- Skipped 'ALL' contour -----")
                    continue
                class_id = CUSTOM_TO_CUBICASA_ID_MAPPER.get(CUBICASA_MAPPER.get(area_class.lower().strip(), DEFAULT_CATEGORY), DEFAULT_CATEGORY)
                #print("Mapped {} to {}".format(area_class.lower().strip(), class_id))
                
                polygon_coords = item["polygon_px"]
                room_polygons.append([polygon_coords, class_id])
        
        
            if image_path.name.endswith(".svg"):
                png_data = cairosvg.svg2png(url=str(image_path))
                img = Image.open(io.BytesIO(png_data)).convert("RGB")
            else:
                img = Image.open(image_path).convert("RGB")
            
            #old_w = data["width"]
            #old_h = data["height"]
            #print("resizing to {}x{}".format(image_size, image_size))
            #img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
            annotated_w = data["width"]
            annotated_h = data["height"]
            #img = img.resize((annotated_w, annotated_h), Image.Resampling.BILINEAR)
            new_w, new_h = img.size
            print(f"Original size: {new_w}, {new_h}")
            print(f"Annotated size: {annotated_w}, {annotated_h}")
            scale_w = new_w / annotated_w
            scale_h = new_h / annotated_h
            print("Scale: {}x{}".format(scale_w, scale_h))

            M = np.float32([
                [scale_w, 0.0, 0.0],
                [0.0, scale_h, 0.0]
            ])

            scaled_room_polygons = []
            for poly, class_name in room_polygons:
                #scaled_poly = []
                #for pt in poly:
                #    x, y = pt
                #    new_x = int(round(x * scale_w))
                #    new_y = int(round(y * scale_h))
                #    #new_x = max(0, min(new_x, image_size - 1))
                #    #new_y = max(0, min(new_y, image_size - 1))
                #    
                #    scaled_poly.append((new_x, new_y))
                original_poly_cv = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                scaled_poly = cv2.transform(original_poly_cv, M).astype(np.int32)
                approx_poly = scaled_poly.astype(np.float32).reshape(-1, 2)
                scaled_room_polygons.append((approx_poly, class_name))
            room_polygons = scaled_room_polygons

            img_array = np.array(img)
            output_image = img_array.copy()
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            inter_image = img_array.copy()
            mask = np.zeros((new_h + 2, new_w + 2), np.uint8)
            cv2.floodFill(output_image, mask, (0, 0), (255, 255, 255))
            cv2.floodFill(inter_image, mask, (0, 0), (255, 255, 255))
            for poly, _ in room_polygons:
                random_color = np.random.randint(0, 256, size=3).tolist()
                original_poly_cv = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.drawContours(output_image, [original_poly_cv], -1, random_color, thickness=3)
            cv2.imwrite("./output_stratify_{}_{}.png".format(scene_id, image_path.stem), output_image)
            
            binary_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            for poly, _ in room_polygons:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(binary_mask, [pts], color=1)

            #filled_mask = fill_holes_in_mask(binary_mask)
            #fixed_mask_size = 512
            #binary_mask_1024 = cv2.resize(
            #    filled_mask,
            #    (fixed_mask_size, fixed_mask_size),
            #    interpolation=cv2.INTER_NEAREST,
            #)
            #img.info.pop("icc_profile", None)
            #if len(binary_mask.shape) == 2 and len(img_array.shape) == 3:
            #    binary_mask = binary_mask[:, :, np.newaxis]

            #if binary_mask_1024 is not None:
            #    if len(binary_mask.shape) == 2 and len(img_array.shape) == 3:
            #        binary_mask_1024 = binary_mask_1024[:, :, np.newaxis]
                #img_1024 = cv2.resize(
                #    img_array,
                #    (fixed_mask_size, fixed_mask_size),
                #    interpolation=cv2.INTER_NEAREST,
                #)
                #img_1024 = cv2.
                #gray_image = cv2.cvtColor(img_1024, cv2.COLOR_BGR2GRAY)
                #optimal_thresh, binary_image = cv2.threshold(
                #    gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                #)
                #binary_3channel = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            #masked_img = np.where(binary_mask == 0, 255, img)
            img = Image.fromarray(inter_image.astype(np.uint8))
            img.save(f"{img_folder}/{str(img_id).zfill(5) + '.png'}")
        
            #coco_annotation_dict_list = []
        
            for poly_ind, (polygon, class_id) in enumerate(room_polygons):  
                poly_shapely = Polygon(polygon)
                area = poly_shapely.area              
                polygon_array = np.array(polygon)
                poly_sorted = resort_corners(polygon_array)

                poly_type = CC5K_MAPPING_2[class_id]
                if poly_type is None:
                    poly_type = DEFAULT_CATEGORY
                #if poly_type not in [10, 9] and area < 100:
                #    continue
                #if poly_type in [10, 9] and area < 1:
                #    continue

                rectangle_shapely = poly_shapely.envelope
                polygon = np.array(polygon)

                coco_seg_poly = []
                poly_sorted = resort_corners(polygon)

                for p in poly_sorted:
                    coco_seg_poly += list(p)
                
                #coco_seg_poly = []
                #for p in poly_sorted:
                #    coco_seg_poly += [int(p[0]), int(p[1])] # was float before

                #pairs = [(coco_seg_poly[i], coco_seg_poly[i+1]) for i in range(0, len(coco_seg_poly), 2)]
                #poly_shapely = Polygon(pairs)
                #area = poly_shapely.area
                
                #if area < 100: 
                #    continue
                    
                # Slightly wider bounding box
                bb_x, bb_y = rectangle_shapely.exterior.xy
                coco_bb = create_coco_bounding_box(bb_x, bb_y, new_w, new_h, bound_pad=2)
                
                #xs = coco_seg_poly[0::2]
                #ys = coco_seg_poly[1::2]
                #x_min, y_min = min(xs), min(ys)
                #width = max(xs) - x_min
                #height = max(ys) - y_min
                #coco_bb = [int(x_min), int(y_min), int(width), int(height)]
                
                coco_annotation_dict = {
                    "segmentation": [coco_seg_poly],
                    "area": int(round(area, 2)),
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": coco_bb,
                    "category_id": poly_type, #int(class_id) - 1,
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

    #print(f"{full_plan}")
    #print(f"Full plan length: {len(full_plan)}")
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

            #class_raw_label = item["raw_label"]
            #if class_raw_label == "All" or class_raw_label.lower() == "all":
            #    print("! ----- Skipped 'ALL' contour -----")
            #    continue
            
            areas = data.get("areas", [])
            keypoints = []
            polygon_raw_ids = []

            for i, item in enumerate(areas):
                raw_id = item["id"]
                poly = item["polygon_px"]
                out_poly = []
                for point in poly:
                    print(point)
                    out_poly.append([point[0], point[1]])  # [x, y]
                    #polygon_raw_ids.append(raw_id) # poly id
                keypoints.append(out_poly)
                polygon_raw_ids.append(tuple([i, raw_id])) # 0: bd_1 etc

            new_w = int(w * 1.5)
            new_h = int(h * 1.5)
            print(f"Original size: {w}, {h}")
            print(f"Aug size: {new_w}, {new_h}")
            scale_w = new_w / w
            scale_h = new_h / h
            print("Scale: {}x{}".format(scale_w, scale_h))

            M = np.float32([
                [scale_w, 0.0, 0.0],
                [0.0, scale_h, 0.0]
            ])

            scaled_keypoints = []
            for poly in keypoints:
                original_poly_cv = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                scaled_poly = cv2.transform(original_poly_cv, M).astype(np.int32)
                approx_poly = scaled_poly.astype(np.float32).reshape(-1, 2).tolist()
                scaled_keypoints.append(approx_poly)

            aug_image = cv2.resize(img_array, (new_w, new_h))

            #img_array = np.array(img)
            #output_image = img_array.copy()
            #output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            #inter_image = img_array.copy()
            ##for poly, _ in room_polygons:
            #   random_color = np.random.randint(0, 256, size=3).tolist()
            #   original_poly_cv = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            #    cv2.drawContours(output_image, [original_poly_cv], -1, random_color, thickness=3)
            #cv2.imwrite("./output_stratify_{}_{}.png".format(scene_id, image_path.stem), output_image)

            #transform = A.Compose([
            #    #A.HorizontalFlip(p=0.5),
            #    #A.RandomRotate90(p=1.0),
            #    A.Resize(height=h*2.0, width=w*2.0, p=1.0),
            #    #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.7),
            #    #A.RandomBrightnessContrast(p=0.5),
            #], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
            #transformed = transform(image=image_cv, keypoints=keypoints)
            #aug_image = transformed['image']
            #aug_keypoints = transformed['keypoints']
            #print(polygon_raw_ids)

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

            #augmented_polygons = [[] for _ in range(len(keypoints))]
            #for (x, y), raw_id in zip(aug_keypoints, polygon_indices):
            #    augmented_polygons[raw_id].append([float(x), float(y)])
            #final_augmented_polygons = [p for p in augmented_polygons if len(p) >= 3]
            #areas[] = final_augmented_polygons


            cv2.imwrite(out_contour_image, image_contours)
            augmented_annotations.append({
                "image": out_aug_image.absolute(),
                "area_file": out_aug_annotation.absolute()
            })
            augmented_images_names.append(out_aug_image.name)

            with open(out_aug_annotation, "w") as f:
                json.dump(data, f)
            #cv2.imwrite(out_aug_image, aug_image)
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
    
    image_size = 512
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

    #k_fold = KFold(n_splits=5, shuffle=True)
    #for fold_idx, (train_index, test_index) in enumerate(k_fold.split(full_plan)):
    #    print(f"--- Fold {fold_idx + 1} ---")
    for i in range(0, 1):
        #print(f"Test items count: {len(test_index)}")
        #print(f"Train items count: {len(train_index)}")

        #train_fold_items = full_plan[train_index]
        #test_fold_items = full_plan[test_index]

        #train_fold_items = [full_plan[i] for i in train_index]
        #test_fold_items = [full_plan[i] for i in test_index]
        train_fold_items = train_dataset
        test_fold_items = test_dataset

        fold_path = os.path.join(out_folder, f"fold_{0}")
        if not os.path.exists(fold_path):
            os.mkdir(fold_path)

        annotation_out_folder = os.path.join(fold_path, "annotations")
        if not os.path.exists(annotation_out_folder):
            os.mkdir(annotation_out_folder)

        train_path = os.path.join(annotation_out_folder, "train.json")
        val_path = os.path.join(annotation_out_folder, "val.json")
        test_path = os.path.join(annotation_out_folder, "test.json")

        img_train_path = os.path.join(fold_path, "train")
        if not os.path.exists(img_train_path):
            os.mkdir(img_train_path)
        
        img_test_path = os.path.join(fold_path, "test")
        if not os.path.exists(img_test_path):
            os.mkdir(img_test_path)

        _prepare_dataset(train_fold_items, image_size, train_path, img_train_path, start_scene_id=start_scene_id)
        _prepare_dataset(test_fold_items, image_size, test_path, img_test_path, start_scene_id=start_scene_id+len(train_fold_items))
        shutil.copy(test_path, val_path)
        print(f"Prepared fold: {fold_path}")
    