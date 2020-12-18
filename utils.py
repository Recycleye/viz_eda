import base64, os, json

def parse_annotations(annotations):
    _, annotations_content = annotations.split(",", 1)
    decoded_annotations = base64.b64decode(annotations_content).decode("UTF-8")
    if not os.path.isdir("./output"):
        os.mkdir("./output")
    path_to_annotations = "./output/annotations.json"
    f = open(path_to_annotations, 'w')
    f.write(decoded_annotations)
    return path_to_annotations

def compute_overview_data(path_to_images, path_to_annotations):
    f = open(path_to_annotations)
    data = json.load(f)

    annotations = data["annotations"]
    images = data["images"]
    categories = data["categories"]
    info = data["info"]

    no_objects = len(annotations)
    no_images = len(images)
    uniform_distribution = 1

    images_by_id = {}
    for image in images:
        images_by_id[image["id"]] = {}
        image_path = image["file_name"]
        image_path = os.path.join(path_to_images, image_path.split('/')[-1])
        images_by_id[image["id"]]["file_name"] = image_path
        images_by_id[image["id"]]["height"] = image["height"]
        images_by_id[image["id"]]["width"] = image["width"]
        images_by_id[image["id"]]["objects"] = []
        images_by_id[image["id"]]["classes"] = []
    images = images_by_id

    classes = {}
    for cat in categories:
        classes[cat["id"]] = {}
        classes[cat["id"]]["name"] = cat["name"]
        classes[cat["id"]]["no_objects"] = 0
        classes[cat["id"]]["min_bbox_height"] = -1
        classes[cat["id"]]["max_bbox_height"] = -1
        classes[cat["id"]]["min_bbox_width"] = -1
        classes[cat["id"]]["max_bbox_width"] = -1
        classes[cat["id"]]["objs_prop"] = 0
        classes[cat["id"]]["imgs_prop"] = 0
        classes[cat["id"]]["unique_imgs_prop"] = 0
        classes[cat["id"]]["images"] = []
        classes[cat["id"]]["unique_images"] = []

    classes_in_annotations = set()

    # Images that are referenced by annotations but do not appear in the dataset
    missing_images = []

    for ann in annotations:
        classes_in_annotations.add(ann["category_id"])
        classes[ann["category_id"]]["no_objects"] += 1
        classes[ann["category_id"]]["images"].append(ann["image_id"])

        if classes[ann["category_id"]]["min_bbox_width"] < 0:
            classes[ann["category_id"]]["min_bbox_width"] = ann["bbox"][2]
        if ann["bbox"][2] < classes[ann["category_id"]]["min_bbox_width"]:
            classes[ann["category_id"]]["min_bbox_width"] = ann["bbox"][2]

        if classes[ann["category_id"]]["max_bbox_width"] < 0:
            classes[ann["category_id"]]["max_bbox_width"] = ann["bbox"][2]
        if ann["bbox"][2] > classes[ann["category_id"]]["max_bbox_width"]:
            classes[ann["category_id"]]["max_bbox_width"] = ann["bbox"][2]

        if classes[ann["category_id"]]["min_bbox_height"] < 0:
            classes[ann["category_id"]]["min_bbox_height"] = ann["bbox"][3]
        if ann["bbox"][3] < classes[ann["category_id"]]["min_bbox_height"]:
            classes[ann["category_id"]]["min_bbox_height"] = ann["bbox"][3]

        if classes[ann["category_id"]]["max_bbox_height"] < 0:
            classes[ann["category_id"]]["max_bbox_height"] = ann["bbox"][3]
        if ann["bbox"][3] > classes[ann["category_id"]]["max_bbox_height"]:
            classes[ann["category_id"]]["max_bbox_height"] = ann["bbox"][3]

        if ann["image_id"] not in images:
            missing_images.append(ann["image_id"])
        else:
            images[ann["image_id"]]["objects"].append([ann])
            images[ann["image_id"]]["classes"].append(ann["category_id"])

    object_lists_lenghts = [len(images[image_id]["objects"]) \
        for image_id in images]
    min_objects_per_image = min(object_lists_lenghts)
    max_objects_per_image = max(object_lists_lenghts)
    avg_objects_per_image = sum(object_lists_lenghts) \
        / len(object_lists_lenghts)
    
    empty_images = []
    for image in images:
        image_objects = set(images[image]["classes"])
        if len(image_objects) == 1:
            class_id = images[image]["classes"][0]
            classes[class_id]["unique_images"].append(image)
        elif len(image_objects) == 0:
            empty_images.append(images[image]["file_name"])
    
    images_with_wrong_dims = [images[image]["file_name"] for image in images \
        if images[image]["height"] != 1920 or image["width"] != 1080]

    for cl in classes:
        classes[cl]["images"] = list(set(classes[cl]["images"]))
        classes[cl]["unique_images"] = list(set(classes[cl]["unique_images"]))
        classes[cl]["objs_prop"] = classes[cl]["no_objects"]*100/no_objects
        classes[cl]["imgs_prop"] = len(classes[cl]["images"])*100/no_images
        classes[cl]["unique_imgs_prop"] = len(classes[cl]["unique_images"])*100/len(classes[cl]["images"])
        if classes[cl]["objs_prop"] < 5 or classes[cl]["objs_prop"] > 80:
            uniform_distribution = 0
    
    class_ids = set(classes.keys())
    missing_classes = list(class_ids.difference(classes_in_annotations))
    missing_classes = [classes[cl]["name"] for cl in missing_classes]

    new_missing_images = []
    for image in missing_images:
        file_name = image[image]["file_name"].split('/')[-1]
        new_missing_images.append(file_name)

    missing_images = new_missing_images

    missing_image_files = []
    for image in images:
        file_name = images[image]["file_name"].split('/')[-1]
        path = os.path.join(path_to_images, file_name)
        if not os.path.isfile(path):
            missing_image_files.append(file_name)

    overview_data = {
        "info" : info,
        "no_objects" : no_objects,
        "no_images" : no_images,
        "uniform_distribution" : uniform_distribution,
        "images" : images,
        "classes" : classes,
        "min_objects_per_image" : min_objects_per_image,
        "max_objects_per_image" : max_objects_per_image,
        "avg_objects_per_image" : avg_objects_per_image,
        "missing_images" : missing_images,
        "empty_images" : empty_images,
        "images_with_wrong_dims" : images_with_wrong_dims,
        "missing_classes" : missing_classes,
        "missing_image_files" : missing_image_files
    }

    if not os.path.isdir("./output"):
        os.mkdir("./output")
    path_to_overview_data = "./output/overview_data.json"
    overview_data_file = open(path_to_overview_data, 'w')
    json.dump(overview_data, overview_data_file)
    return path_to_overview_data
