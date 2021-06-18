import base64
import json
import os
from statistics import stdev, mean


def analyze_dataset(images_path, anns_path):
    """
    Computes and stores the analysis of the dataset

    :param images_path: path to images
    :param anns_path: path to annotations
    :return: path to analysis file
    """
    f = open(anns_path)
    data = json.load(f)
    info = data["info"]
    all_anns = data["annotations"]
    all_images = data["images"]
    all_classes = data["categories"]

    ###########################################################################
    # Check class IDs

    # Step 1: get all class IDs
    class_ids = []
    class_ids_set = set()
    for cl in all_classes:
        class_ids.append(int(cl["id"]))
        # unique category id
        class_ids_set.add(int(cl["id"]))

    # Step 2: define range of possible IDs [min_id,max_id]
    min_id = min(class_ids)
    max_id = max(class_ids)
    ids_range = str(min_id) + "-" + str(max_id)

    # Step 3: count occurrences of each ID
    # id_count contains a list of (max+id+1) 0s, category_id starts from 0
    id_count = [0] * (max_id + 1)
    for class_id in class_ids:
        id_count[class_id] += 1

    # Step 3: check no ID is used more than once and that there are no gaps
    unused_ids = []
    repeated_ids = []
    for i in range(len(id_count)):
        if id_count[i] == 0:
            unused_ids.append(i)
        elif id_count[i] > 1:
            repeated_ids.append(i)

    ###########################################################################
    # Order images and annotations by class

    classes = {}
    for cl in all_classes:
        # Order by ID
        cl_id = int(cl["id"])
        classes[cl_id] = {}

        # dicts in dicts
        classes[cl_id]["name"] = cl["name"].lower()
        classes[cl_id]["num_objects"] = 0
        classes[cl_id]["num_images"] = 0
        classes[cl_id]["images"] = set()
        classes[cl_id]["size_avg"] = {
            "area": [],
            "ann": [],
            "avg": 0,
            "std": 0
        }
        classes[cl_id]["bbox_min_dims"] = {
            "width": 10000000,
            "height": 10000000
        }
        classes[cl_id]["bbox_max_dims"] = {
            "width": -1,
            "height": -1
        }


    ###########################################################################
    # Get all images info

    wrong_dims = set()
    image_ids_set = set()
    images = {}
    for image in all_images:
        # Order by ID
        image_ids_set.add(int(image["id"]))
        image_id = int(image["id"])
        images[image_id] = {}

        # Get the file name
        file_name = image["file_name"].split('/')[-1]
        file_name = os.path.join(images_path, file_name)
        images[image_id]["file_name"] = file_name

        # Get the image dimensions width x height
        images[image_id]["width"] = image["width"]
        images[image_id]["height"] = image["height"]
        if image["width"] != 1920 or image["height"] != 1080:
            wrong_dims.add(image_id)
        # Initialise empty lists for all objects and classes in image
        images[image_id]["objects"] = []
        images[image_id]["classes"] = set()

    ###########################################################################
    # Check the classes and images referenced by the annotations

    # Step 1: collect all class and image IDs referenced in anns
    referenced_classes = set()
    referenced_images = set()
    # all_anns: annotations
    for ann in all_anns:
        referenced_classes.add(int(ann["category_id"]))
        referenced_images.add(int(ann["image_id"]))

    # Step 2: get empty class and image ID
    empty_classes = list(class_ids_set.difference(referenced_classes))
    empty_images = list(image_ids_set.difference(referenced_images))

    # Step 3: get missing class and image IDS 
    missing_classes = list(referenced_classes.difference(class_ids_set))
    missing_images = list(referenced_images.difference(image_ids_set))

    ###########################################################################
    # Add annotations to images, images to classes, and update bbox dims

    min_bbox_dims_string = "10000000x10000000"
    min_bbox_dims = 10000000 * 10000000
    min_bbox_dims_class = ""
    max_bbox_dims_string = "-1*1"
    max_bbox_dims = -1 * 1
    max_bbox_dims_class = ""

    for ann in all_anns:

        ann_id = ann["id"]

        # Get the class ID
        class_id = int(ann["category_id"])

        # Update objects count in class dict
        classes[class_id]["num_objects"] += 1

        # Get the image ID
        image_id = int(ann["image_id"])

        # Add annotation to image
        images[image_id]["objects"].append(ann)
        # Add class ID to image
        images[image_id]["classes"].add(class_id)

        # Add image to class
        classes[class_id]["images"].add(image_id)

        # Update bbox dims
        prev_min_width = classes[class_id]["bbox_min_dims"]["width"]
        prev_max_width = classes[class_id]["bbox_max_dims"]["width"]
        curr_width = int(ann["bbox"][2])
        classes[class_id]["bbox_min_dims"]["width"] = min(prev_min_width, curr_width)
        classes[class_id]["bbox_max_dims"]["width"] = max(prev_max_width, curr_width)

        prev_min_height = classes[class_id]["bbox_min_dims"]["height"]
        prev_max_height = classes[class_id]["bbox_max_dims"]["height"]
        curr_height = int(ann["bbox"][3])
        classes[class_id]["bbox_min_dims"]["height"] = min(prev_min_height, curr_height)
        classes[class_id]["bbox_max_dims"]["height"] = max(prev_max_height, curr_height)

        curr_dims = curr_width*curr_height
        curr_area = ann["area"]
        classes[class_id]["size_avg"]["area"].append(curr_area)
        classes[class_id]["size_avg"]["ann"].append(ann_id)
        if curr_dims < min_bbox_dims:
            min_bbox_dims_string = str(ann["bbox"][2]) + "x" + str(ann["bbox"][3])
            min_bbox_dims = curr_dims
            min_bbox_dims_class = classes[class_id]["name"]
        if curr_dims > max_bbox_dims:
            max_bbox_dims_string = str(ann["bbox"][2]) + "x" + str(ann["bbox"][3])
            max_bbox_dims = curr_dims
            max_bbox_dims_class = classes[class_id]["name"]

    for cl in all_classes:
        cl_id = int(cl["id"])
        classes[cl_id]["size_avg"]['avg'] = mean(classes[cl_id]["size_avg"]['area'])
        classes[cl_id]["size_avg"]['std'] = stdev(classes[cl_id]["size_avg"]['area'])

    bbox_stats = {
        "min": min_bbox_dims_string,
        "min_class": min_bbox_dims_class,
        "max": max_bbox_dims_string,
        "max_class": max_bbox_dims_class
    }

    ###########################################################################
    # Update images count per class

    for class_id in classes:
        num_images = len(classes[class_id]["images"])
        classes[class_id]["num_images"] = num_images

    ###########################################################################
    # Get min, max and avg objects per image

    min_objects_per_image = 10000000
    max_objects_per_image = -1
    avg_objects_per_image = 0
    for image in images:
        curr_num_objects = len(images[image]["objects"])
        min_objects_per_image = min(min_objects_per_image, curr_num_objects)
        max_objects_per_image = max(max_objects_per_image, curr_num_objects)
        avg_objects_per_image += curr_num_objects

    avg_objects_per_image /= len(all_images)

    objects_per_image_stats = {
        "min": str(min_objects_per_image),
        "max": str(max_objects_per_image),
        "avg": str(avg_objects_per_image)
    }
    ###########################################################################
    # Change all sets into lists to allow JSON serialization

    for cl in classes:
        classes[cl]["images"] = list(classes[cl]["images"])

    for image in images:
        images[image]["classes"] = list(images[image]["classes"])

    wrong_dims = list(wrong_dims)

    ###########################################################################
    # Group analysis results together, write to file and return path

    analysis = {
        "images_path": images_path,
        "annotation_path": anns_path,
        "info": info,
        "total_num_objects": str(len(all_anns)),
        "total_num_images": str(len(all_images)),
        "classes": classes,
        "images": images,
        "objects_per_image_stats": objects_per_image_stats,
        "bbox_stats": bbox_stats,
        "ids_range": ids_range,
        "unused_IDs": unused_ids,
        "repeated_IDs": repeated_ids,
        "empty_classes": empty_classes,
        "empty_images": empty_images,
        "missing_classes": missing_classes,
        "missing_images": missing_images,
        "wrong_dims": wrong_dims
    }
    # usage of os.path.isdir: https://www.geeksforgeeks.org/python-os-path-isdir-method/
    if not os.path.isdir("./output"):
        os.mkdir("./output")
    analysis_path = "./output/analysis.json"
    # open mode: w means Write - Opens a file for writing, creates the file if it does not exist
    f = open(analysis_path, 'w')
    # json.dumps: convert from python to json https://www.w3schools.com/python/python_json.asp
    data = json.dumps(analysis, indent=4)
    f.write(data)

    return analysis_path

def parse_annotations(content):
    """
    Parses the annotations and writes them to file for later use

    :param contents: the contents of the upload btn
    :return: path to annotations file
    """
    # base64decode: https://docs.python.org/3/library/base64.html
    # Decode the Base64 encoded bytes-like object or ASCII string s and return the decoded bytes.
    decoded_annotations = base64.b64decode(content).decode("UTF-8")
    if not os.path.isdir("./output"):
        os.mkdir("./output")
    path_to_annotations = "./output/annotations.json"
    f = open(path_to_annotations, 'w')
    f.write(decoded_annotations)
    return path_to_annotations


# not used
def parse_analysis(content):
    """
    Parses the analysis and writes it to file for later use

    :param contents: the contents of the upload btn
    :return: path to analysis file
    """
    decoded_analysis = base64.b64decode(content).decode("UTF-8")
    if not os.path.isdir("./output"):
        os.mkdir("./output")
    path_to_analysis = "./output/analysis.json"
    f = open(path_to_analysis, 'w')
    f.write(decoded_analysis)
    return path_to_analysis
