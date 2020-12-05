import json 
import os

def compute_overview_data(path_to_images, path_to_annotations):

    f = open(path_to_annotations)
    data = json.load(f)

    annotations = data["annotations"]
    images = data["images"]
    categories = data["categories"]
    info = data["info"]

    classes = {}
    for cat in categories:
        classes[cat["id"]] = {}
        classes[cat["id"]]["name"]  = cat["name"]
        classes[cat["id"]]["anns_count"] = 0
        classes[cat["id"]]["anns_prop"] = 0
        classes[cat["id"]]["imgs_count"] = 0
        classes[cat["id"]]["imgs_prop"] = 0
        classes[cat["id"]]["unique_imgs"] = []
        classes[cat["id"]]["unique_imgs_prop"] = 0
        classes[cat["id"]]["imgs"] = []
        classes[cat["id"]]["anns"] = []
    
    anns_per_img = {}
    classes_in_anns = set()
    for ann in annotations:
        classes_in_anns.add(ann["category_id"])
        classes[ann["category_id"]]["anns"].append(ann)
        classes[ann["category_id"]]["imgs"].append(ann["image_id"])
        if ann["image_id"] not in anns_per_img:
            anns_per_img[ann["image_id"]] = []
        anns_per_img[ann["image_id"]].append(ann["category_id"])

    anns_per_img_ids = list(anns_per_img.keys())
    anns_per_img_count = [len(anns_list) for anns_list \
        in list(anns_per_img.values())]
    min_anns_per_img = min(anns_per_img_count)
    max_anns_per_img = max(anns_per_img_count)
    avg_anns_per_img = sum(anns_per_img_count) / len(anns_per_img_count)

    for img in anns_per_img:
        if len(set(anns_per_img[img])) == 1:
            cl_id = anns_per_img[img][0]
            classes[cl_id]["unique_imgs"].append(img)
        
    img_ids = [img["id"] for img in images]
    imgs_with_no_anns = list(set(img_ids).difference(set(anns_per_img_ids)))
    anns_with_no_imgs = list(set(anns_per_img_ids).difference(set(img_ids)))

    imgs_wrong_dims = [img["file_name"] for img in images \
        if img["height"]!=1920 or img["width"]!=1080] 
    
    for cl in classes:
        classes[cl]["anns_count"] = len(classes[cl]["anns"])
        classes[cl]["anns_prop"] = classes[cl]["anns_count"]*100 \
            / len(annotations)
        classes[cl]["imgs_count"] = len(set(classes[cl]["imgs"]))
        classes[cl]["imgs_prop"] = classes[cl]["imgs_count"]*100 / len(images)
        classes[cl]["unique_imgs_count"] = len(set(classes[cl]["unique_imgs"]))
        classes[cl]["unique_imgs_prop"] = classes[cl]["unique_imgs_count"]*100 \
            / classes[cl]["imgs_count"]
        img_files = [img["file_name"] for img in images \
            if img["id"] in classes[cl]["imgs"]]
        classes[cl]["imgs"] = img_files

    class_proportions = [classes[cl]["anns_prop"] for cl in classes]
    uniform_distribution = 1
    for prop in class_proportions:
        if prop < 5 or prop > 80:
            uniform_distribution = 0

    classes_ids = set(list(classes.keys()))
    missing_classes = list(classes_ids.difference(classes_in_anns))

    missing_imgs = []
    for img in images:
        file_name = img["file_name"].split('/')[-1]
        img_path = os.path.join(path_to_images, file_name)
        if not os.path.isfile(img_path):
            missing_imgs.append(file_name)

    overview_data = {
        "info": info,
        "classes" : classes,
        "uniform_distribution" : uniform_distribution,
        "missing_classes" : missing_classes,
        "anns_count" : len(annotations),
        "imgs_count" : len(images),
        "min_anns_per_img" : min_anns_per_img,
        "max_anns_per_img" : max_anns_per_img,
        "avg_anns_per_img" : avg_anns_per_img,
        "imgs_with_no_anns" : imgs_with_no_anns,
        "anns_with_no_imgs" : anns_with_no_imgs,
        "imgs_wrong_dims" : imgs_wrong_dims,
        "missing_imgs" : missing_imgs
    }

    return overview_data