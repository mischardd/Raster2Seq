import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loaders import FloorplanSVG
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

#### ORIGINAL ROOM NAMES & ICON_NAMES ####
ROOM_NAMES = {
    0: "Background",
    1: "Outdoor",
    2: "Wall",
    3: "Kitchen",
    4: "Living Room",
    5: "Bed Room",
    6: "Bath",
    7: "Entry",
    8: "Railing",
    9: "Storage",
    10: "Garage",
    11: "Undefined",
}

ICON_NAMES = {
    0: "No Icon",
    1: "Window",
    2: "Door",
    3: "Closet",
    4: "Electrical Applience",
    5: "Toilet",
    6: "Sink",
    7: "Sauna Bench",
    8: "Fire Place",
    9: "Bathtub",
    10: "Chimney",
}


CC5K_2_S3D_MAPPING = {
    0: None,  # "Background"
    1: type2id["balcony"],  # "Outdoor" -> balcony (4)
    2: None,  # "Wall" has no direct match
    3: type2id["kitchen"],  # Kitchen -> kitchen (1)
    4: type2id["living room"],  # Living Room -> living room (0)
    5: type2id["bedroom"],  # Bed Room -> bedroom (2)
    6: type2id["bathroom"],  # Bath -> bathroom (3)
    7: 18,  # 'Entry' has no direct match
    8: 19,  # "Railing" has no direct match
    9: type2id["store room"],  # Storage -> store room (9)
    10: type2id["garage"],  # Garage -> garage (14)
    11: type2id["undefined"],  # Undefined -> undefined (15)
    12: type2id["window"],  # Window -> window (17)
    13: type2id["door"],  # Door -> door (16)
}

CC5K_MAPPING = {
    0: None,
    1: 0,  # Outdoor
    2: 1,  # Wall
    3: 2,  # Kitchen
    4: 3,  # Living Room
    5: 4,  # Bed Room
    6: 5,  # Bath
    7: 6,  # Entry
    8: 1,  # Railing -> Wall
    9: 7,  # Storage
    10: 8,  # Garage
    11: 9,  # Undefined
    12: 10,  # Window
    13: 11,  # Door
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

CC5K_CLASS_MAPPING = {
    "Outdoor": 0,
    "Wall, Railing": 1,
    "Kitchen": 2,
    "Living Room": 3,
    "Bed Room": 4,
    "Bath": 5,
    "Entry": 6,
    "Storage": 7,
    "Garage": 8,
    "Undefined": 9,
    "Window": 10,
    "Door": 11,
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

CLASS_MAPPING = {
    "living room": 0,
    "kitchen": 1,
    "bedroom": 2,
    "bathroom": 3,
    "balcony": 4,
    "corridor": 5,
    "dining room": 6,
    "study": 7,
    "studio": 8,
    "store room": 9,
    "garden": 10,
    "laundry room": 11,
    "office": 12,
    "basement": 13,
    "garage": 14,
    "undefined": 15,
    "door": 16,
    "window": 17,
    "entry": 18,
    "railing": 19,
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
    a = argparse.ArgumentParser(description="Generate coco format data for Structured3D")
    a.add_argument(
        "--data_root", default="Structured3D_panorama", type=str, help="path to raw Structured3D_panorama folder"
    )
    a.add_argument("--output", default="coco_cubicasa5k", type=str, help="path to output folder")
    a.add_argument("--disable_wd2line", action="store_true")

    args = a.parse_args()
    return args


def save_image(image_path, output_path, mask=None):
    """
    ref: https://github.com/ultralytics/ultralytics/issues/339
    """
    img = Image.open(image_path).convert("RGB")
    img.info.pop("icc_profile", None)

    if mask is not None:
        img_array = np.array(img)
        if len(mask.shape) == 2 and len(img_array.shape) == 3:
            mask = mask[:, :, np.newaxis]
        masked_img = np.where(mask == 0, 255, img_array)
        img = Image.fromarray(masked_img.astype(np.uint8))

    img.save(output_path)


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


def process_floorplan(
    image_set,
    scene_id,
    start_scene_id,
    args,
    save_dir,
    annos_folder,
    use_org_cc5k_classs=False,
    vis_fp=False,
    wd2line=False,
):
    if use_org_cc5k_classs:
        class_mapping_dict = CC5K_MAPPING_2  # old: CC5K_MAPPING
        class_to_index_dict = CC5K_CLASS_MAPPING_2
        door_window_index = [10, 9]
    else:
        class_mapping_dict = CC5K_2_S3D_MAPPING
        class_to_index_dict = CLASS_MAPPING
        door_window_index = [16, 17]

    mask = image_set["label"].numpy()
    room_polygons = [[poly, poly_type] for poly, poly_type in zip(image_set["room_polygon"], image_set["room_type"])]
    icon_polygons = [[poly, poly_type] for poly, poly_type in zip(image_set["icon_polygon"], image_set["icon_type"])]

    image_height, image_width = mask.shape[1:]
    coco_annotation_dict_list = []

    # for storing
    save_dict = prepare_dict(class_to_index_dict)  # old: CC5K_CLASS_MAPPING

    instance_id = 0
    img_id = int(scene_id) + start_scene_id
    img_dict = {}
    img_dict["file_name"] = str(img_id).zfill(5) + ".png"
    img_dict["id"] = img_id
    img_dict["width"] = image_width
    img_dict["height"] = image_height

    if vis_fp:
        os.makedirs(save_dir.rstrip("/") + "_aux", exist_ok=True)
        visualize_room_polygons(
            mask[0],
            room_polygons,
            list(ROOM_NAMES.values()),
            save_path=f"{save_dir.rstrip('/') + '_aux'}/{str(img_id).zfill(5)}_room.png",
        )
        visualize_room_polygons(
            mask[1],
            icon_polygons,
            list(ICON_NAMES.values()),
            bg_polygons=room_polygons,
            save_path=f"{save_dir.rstrip('/') + '_aux'}/{str(img_id).zfill(5)}_icon.png",
        )

    #### FILTER NON-USE TYPES
    # DROP BG
    room_skip_types = [0]
    filtered_room_polygons = remove_polygons_by_type(room_polygons, skip_types=room_skip_types)
    # visualize_room_polygons(mask[0], filtered_room_polygons, list(ROOM_NAMES.values()),
    #                         save_path=f"{save_dir.rstrip('/') + '_aux'}/{str(img_id).zfill(5)}_room_filtered.png")

    # Exclude all furnitures, excepts window, door
    icon_skip_types = [0, *list(range(3, 11))]
    filtered_icon_polygons = remove_polygons_by_type(icon_polygons, skip_types=icon_skip_types)
    # visualize_room_polygons(mask[1], filtered_icon_polygons, list(ICON_NAMES.values()),
    #                         bg_polygons=room_polygons, save_path=f"{save_dir.rstrip('/') + '_aux'}/{str(img_id).zfill(5)}_icon_filtered.png")

    #### COMBINED
    combined_polygons = merge_rooms_and_icons(filtered_room_polygons, filtered_icon_polygons)

    filtered_mask1 = mask[0].copy()
    filtered_mask1[np.isin(mask[0], room_skip_types)] = 0

    filtered_mask2 = mask[1].copy()
    filtered_mask2[np.isin(mask[1], icon_skip_types)] = 0
    filtered_mask2[filtered_mask2 != 0] += 11

    filtered_mask = np.where(filtered_mask2 != 0, filtered_mask2, filtered_mask1)

    new_filtered_mask = filtered_mask.copy()
    for src_type, dest_type in class_mapping_dict.items():
        if dest_type is None:
            continue
        new_filtered_mask[filtered_mask == src_type] = dest_type + 1
    # filtered_mask = new_filtered_mask

    binary_mask = np.zeros_like(filtered_mask)
    binary_mask = np.where((mask[0] + mask[1]) != 0, 1, 0).astype(np.uint8)
    filled_mask = fill_holes_in_mask(binary_mask)
    cv2.imwrite(
        f"{save_dir.rstrip('/') + '_aux'}/{str(img_id).zfill(5) + '_mask.png'}", filled_mask.astype(np.uint8) * 255
    )
    # visualize_room_polygons(combined_mask, combined_polygons, list(ROOM_NAMES.values()) + list(ICON_NAMES.values()), save_path=f"{save_dir}/{str(img_id).zfill(5)}_combined.png")

    save_image(
        f"{args.data_root}/{image_set['folder']}/F1_scaled.png",
        f"{save_dir}/{str(img_id).zfill(5) + '.png'}",
        mask=filled_mask,
    )
    if vis_fp:
        save_image(
            f"{args.data_root}/{image_set['folder']}/F1_scaled.png",
            f"{save_dir.rstrip('/') + '_aux'}/{str(img_id).zfill(5) + '_org.png'}",
            mask=None,
        )

    output_polygon_list = []
    combined_polygon_list = []
    for poly_ind, (polygon, poly_type) in enumerate(combined_polygons):
        poly_shapely = Polygon(polygon)
        area = poly_shapely.area

        org_poly_type = poly_type
        poly_type = class_mapping_dict[poly_type]
        if poly_type is None:
            continue

        if poly_type not in door_window_index and area < 100:
            continue
        if poly_type in door_window_index and area < 1:
            continue

        rectangle_shapely = poly_shapely.envelope
        polygon = np.array(polygon)

        ### here we convert door/window annotation into a single line
        if poly_type in door_window_index and wd2line:
            if polygon.shape[0] > 4:
                if len(polygon) == 5 and (polygon[0] == polygon[-1]).all():
                    polygon = polygon[:-1]  # drop last point since it is same as first
                else:
                    bounding_rect = np.array(poly_shapely.minimum_rotated_rectangle.exterior.coords)
                    polygon = bounding_rect[:4]

            assert polygon.shape[0] == 4
            midp_1 = (polygon[0] + polygon[1]) / 2
            midp_2 = (polygon[1] + polygon[2]) / 2
            midp_3 = (polygon[2] + polygon[3]) / 2
            midp_4 = (polygon[3] + polygon[0]) / 2

            dist_1_3 = np.square(midp_1 - midp_3).sum()
            dist_2_4 = np.square(midp_2 - midp_4).sum()
            if dist_1_3 > dist_2_4:
                polygon = np.row_stack([midp_1, midp_3])
            else:
                polygon = np.row_stack([midp_2, midp_4])

        coco_seg_poly = []
        poly_sorted = resort_corners(polygon)

        for p in poly_sorted:
            coco_seg_poly += list(p)

        # Slightly wider bounding box
        bb_x, bb_y = rectangle_shapely.exterior.xy
        coco_bb = create_coco_bounding_box(bb_x, bb_y, image_width, image_height, bound_pad=2)

        coco_annotation_dict = {
            "segmentation": [coco_seg_poly],
            "area": area,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": coco_bb,
            "category_id": poly_type,
            "id": instance_id,
        }
        coco_annotation_dict_list.append(coco_annotation_dict)
        instance_id += 1

        combined_polygon_list.append([np.array(coco_seg_poly).reshape(-1, 2), org_poly_type])
        output_polygon_list.append([np.array(coco_seg_poly).reshape(-1, 2), poly_type + 1])

    #### end split_file loop
    save_dict["images"].append(img_dict)
    save_dict["annotations"] += coco_annotation_dict_list

    json_path = f"{annos_folder}/{str(img_id).zfill(5) + '.json'}"
    with open(json_path, "w") as f:
        json.dump(save_dict, f)

    if vis_fp:
        visualize_room_polygons(
            filtered_mask,
            combined_polygon_list,
            list(ROOM_NAMES.values()) + ["window", "door"],
            save_path=f"{save_dir.rstrip('/') + '_aux'}/{str(img_id).zfill(5)}_combined.png",
        )
        visualize_room_polygons(
            new_filtered_mask,
            output_polygon_list,
            ["null"] + list(class_to_index_dict.keys()),
            save_path=f"{save_dir.rstrip('/') + '_aux'}/{str(img_id).zfill(5)}_final.png",
        )


def prepare_dict(categories_dict):
    save_dict = {"images": [], "annotations": [], "categories": []}
    for key, value in categories_dict.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        save_dict["categories"].append(type_dict)
    return save_dict


if __name__ == "__main__":
    args = config()

    ### prepare
    outFolder = args.output
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    annotation_outFolder = os.path.join(outFolder, "annotations_json")
    if not os.path.exists(annotation_outFolder):
        os.mkdir(annotation_outFolder)

    annos_train_folder = os.path.join(annotation_outFolder, "train")
    annos_val_folder = os.path.join(annotation_outFolder, "val")
    annos_test_folder = os.path.join(annotation_outFolder, "test")
    os.makedirs(annos_train_folder, exist_ok=True)
    os.makedirs(annos_val_folder, exist_ok=True)
    os.makedirs(annos_test_folder, exist_ok=True)

    train_img_folder = os.path.join(outFolder, "train")
    val_img_folder = os.path.join(outFolder, "val")
    test_img_folder = os.path.join(outFolder, "test")

    for img_folder in [train_img_folder, val_img_folder, test_img_folder]:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

    coco_train_json_path = os.path.join(annotation_outFolder, "train.json")
    coco_val_json_path = os.path.join(annotation_outFolder, "val.json")
    coco_test_json_path = os.path.join(annotation_outFolder, "test.json")

    ### begin processing
    start_scene_id = 3500  # following index of s3d data
    split_set = ["train.txt", "val.txt", "test.txt"]
    save_folders = [train_img_folder, val_img_folder, test_img_folder]
    coco_json_paths = [coco_train_json_path, coco_val_json_path, coco_test_json_path]
    annos_folders = [annos_train_folder, annos_val_folder, annos_test_folder]

    def wrapper(scene_id):
        image_set = dataset[scene_id]
        process_floorplan(
            image_set,
            scene_id,
            start_scene_id,
            args,
            save_dir,
            annos_folder,
            use_org_cc5k_classs=True,
            vis_fp=scene_id < 100,
            wd2line=not args.disable_wd2line,
        )

    def worker_init(dataset_obj):
        # Store dataset as global to avoid pickling issues
        global dataset
        dataset = dataset_obj

    for split_id, split_file in enumerate(split_set):
        dataset = FloorplanSVG(args.data_root, split_file, format="txt", original_size=False)
        save_dir = save_folders[split_id]
        json_path = coco_json_paths[split_id]
        print(f"############# {split_file}")

        annos_folder = annos_folders[split_id]
        num_processes = 1
        with Pool(num_processes, initializer=worker_init, initargs=(dataset,)) as p:
            indices = range(len(dataset))
            list(tqdm(p.imap(wrapper, indices), total=len(dataset)))

        start_scene_id += len(dataset)
