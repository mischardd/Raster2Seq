from .poly_data import build as build_poly


def build_dataset(image_set, args):
    if args.dataset_name in ["stru3d", "cubicasa", "waffle", "r2g", "custom"]:
        print(f"Build {args.dataset_name} {image_set} dataset")
        return build_poly(image_set, args)
    raise ValueError(f"dataset {args.dataset_name} not supported")


def get_dataset_class_labels(dataset_name):
    semantics_label = None

    if dataset_name == "stru3d":
        semantics_label = {
            0: "Living Room",
            1: "Kitchen",
            2: "Bedroom",
            3: "Bathroom",
            4: "Balcony",
            5: "Corridor",
            6: "Dining room",
            7: "Study",
            8: "Studio",
            9: "Store room",
            10: "Garden",
            11: "Laundry room",
            12: "Office",
            13: "Basement",
            14: "Garage",
            15: "Misc.",
            16: "Door",
            17: "Window",
        }
    elif dataset_name == "cubicasa":
        semantics_label = {
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
    elif dataset_name == "r2g":
        semantics_label = {
            "unknown": 0,
            "living_room": 1,
            "kitchen": 2,
            "bedroom": 3,
            "bathroom": 4,
            "restroom": 5,
            "balcony": 6,
            "closet": 7,
            "corridor": 8,
            "washing_room": 9,
            "PS": 10,
            "outside": 11,
        }
    elif dataset_name == "custom":
        semantics_label = {
            "livingroom": 1,
            "bedroom": 2,
            "kitchen": 3,
            "bathroom": 4,
            "corridor": 5,
            "balcony": 6,
            "elevator": 7,
            "stairs": 8,
            "utilityroom": 9,
            "closet": 10,
            "courtyard": 11,
            "entry": 12
        }

    id2class = {v: k for k, v in semantics_label.items()} if semantics_label else None

    return semantics_label, id2class
