import numpy as np
import pandas as pd
import cv2
import random
from pycocotools import coco as coco
from tqdm import tqdm
from anomaly import getOutliers, getAnomalies
import time


def get_num_objs(filter_classes, coco_data):
    # Returns number of objects of a given class
    cat_ids = coco_data.getCatIds(catNms=filter_classes)
    ann_ids = coco_data.getAnnIds(catIds=cat_ids)
    return len(ann_ids)


def get_num_imgs(filter_classes, coco_data):
    # Returns number of imgs with an object of a given class
    cat_ids = coco_data.getCatIds(catNms=filter_classes)
    img_ids = coco_data.getImgIds(catIds=cat_ids)
    return len(img_ids)


def get_objs_per_img(filter_classes, coco_data):
    # Returns average number of objects of a given class in an image
    cat_ids = coco_data.getCatIds(catNms=filter_classes)
    img_ids = coco_data.getImgIds(catIds=cat_ids)

    data = {}
    for img in img_ids:
        ann_ids = coco_data.getAnnIds(imgIds=img, catIds=cat_ids)
        data[len(data)] = [img, len(ann_ids)]
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["imgID", "number of objs"]
    )
    avg = df["number of objs"].sum() / len(df["number of objs"])
    return avg, df


def get_proportion(filter_classes, coco_data, ann_ids=None):
    # Returns average proportion an object of a given class takes up in the image
    cat_ids = coco_data.getCatIds(catNms=filter_classes)
    img_ids = coco_data.getImgIds(catIds=cat_ids)

    data = {}
    for imgId in img_ids:
        im_ann = coco_data.loadImgs(ids=imgId)[0]
        all_ann_ids = coco_data.getAnnIds(imgIds=imgId, catIds=cat_ids, iscrowd=0)
        if ann_ids is not None:
            all_ann_ids = list(set(all_ann_ids).intersection(set(ann_ids)))
        objs = coco_data.loadAnns(ids=all_ann_ids)
        for obj in objs:
            if "area" in obj:
                proportion = obj["area"] / (im_ann["width"] * im_ann["height"])
            else:
                poly_verts = segment_to_2d_array(obj["segmentation"])
                area = get_area(poly_verts)
                proportion = area / (im_ann["width"] * im_ann["height"])
            data[len(data)] = [imgId, obj["id"], proportion]
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["imgID", "annID", "proportion of img"]
    )
    avg = df["proportion of img"].sum() / len(df["proportion of img"])
    return avg, df


def get_seg_roughness(filter_classes, coco_data, ann_ids=None):
    # Returns average roughness an object of a given class
    # "Roughness" = number of segmentation vertices / area of obj
    cat_ids = coco_data.getCatIds(catNms=filter_classes)
    img_ids = coco_data.getImgIds(catIds=cat_ids)

    data = {}
    for imgID in img_ids:
        all_ann_ids = coco_data.getAnnIds(imgIds=imgID, catIds=cat_ids, iscrowd=0)
        if ann_ids is not None:
            all_ann_ids = list(set(all_ann_ids).intersection(set(ann_ids)))
        objs = coco_data.loadAnns(ids=all_ann_ids)
        for obj in objs:
            num_vertices = len(obj["segmentation"])
            if "area" in obj:
                roughness = num_vertices / obj["area"]
            else:
                poly_verts = segment_to_2d_array(obj["segmentation"])
                area = get_area(poly_verts)
                roughness = (num_vertices * 1000) / area
            data[len(data)] = [imgID, obj["id"], roughness]
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["imgID", "annID", "roughness of annotation"]
    )
    avg = df["roughness of annotation"].sum() / len(df["roughness of annotation"])
    return avg, df


def get_area(polygon):
    xs = np.array([x[0] for x in polygon])
    ys = np.array([y[1] for y in polygon])
    return 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def segment_to_2d_array(segmentation):
    polygon = []
    for partition in segmentation:
        for x, y in zip(partition[::2], partition[1::2]):
            polygon.append((x, y))
    return polygon


def mask_pixels(polygon, img_dict, image_folder):
    img = cv2.imread("{}/{}".format(image_folder, img_dict["file_name"]))
    mask = np.zeros(img.shape, dtype=np.uint8)
    polygon = np.int32(polygon)
    mask = cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
    return img, mask


def get_segmented_masks(filter_classes, image_folder, coco_data):
    # Returns single object annotation with black background
    cat_ids = coco_data.getCatIds(catNms=filter_classes)
    img_ids = coco_data.getImgIds(catIds=cat_ids)

    if len(img_ids) > 500:
        img_ids = random.sample(img_ids, 500)
    imgs = coco_data.loadImgs(img_ids)

    mask_ann_ids = []
    masks = []
    loaded_imgs = []
    for img_dict in tqdm(imgs):
        # Load annotations
        ann_ids = coco_data.getAnnIds(imgIds=img_dict["id"], catIds=cat_ids, iscrowd=0)
        anns = coco_data.loadAnns(ann_ids)
        mask_ann_ids.extend(ann_ids)
        # Create masked images
        for ann in anns:
            poly_verts = segment_to_2d_array(ann["segmentation"])
            img, mask = mask_pixels(poly_verts, img_dict, image_folder)
            masks.append(mask)
            loaded_imgs.append(img)
    return loaded_imgs, masks, mask_ann_ids


def get_histograms(images, masks):
    data = []
    for image, mask in tqdm(zip(images, masks)):
        img_float32 = np.float32(image)
        bgr_planes = cv2.split(img_float32)
        hist_size = 3
        hist_range = (0, 256)  # the upper boundary is exclusive
        accumulate = False

        b_hist = cv2.calcHist(
            bgr_planes, [0], mask, [hist_size], hist_range, accumulate=accumulate
        )
        g_hist = cv2.calcHist(
            bgr_planes, [1], mask, [hist_size], hist_range, accumulate=accumulate
        )
        r_hist = cv2.calcHist(
            bgr_planes, [2], mask, [hist_size], hist_range, accumulate=accumulate
        )

        hist_h = 400
        b_features = cv2.normalize(
            b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX
        ).flatten()
        g_features = cv2.normalize(
            g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX
        ).flatten()
        r_features = cv2.normalize(
            r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX
        ).flatten()
        data.append([b_features, g_features, r_features])
    return np.array(data)


def get_obj_colors(images, masks):
    n_colors = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    data = []
    img_masks = zip(images, masks)
    for image, mask in tqdm(img_masks):
        masked_image = cv2.bitwise_and(image, mask)
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

        pixels = np.float32(masked_image.reshape(-1, 3))
        pixels = pixels[np.all(pixels != 0, axis=1), :]
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)

        indices = np.argsort(counts)[::-1]
        dom_colour = palette[indices[0]]
        data.append(dom_colour)
    return data


def analyze_dataset(annotation_file, image_folder):
    coco_data = coco.COCO(annotation_file)
    cats = coco_data.loadCats(coco_data.getCatIds())
    nms = [cat["name"] for cat in cats]

    data = {}
    cat_num = 1
    for cat in nms:
        print(cat + ": " + str(cat_num) + "/" + str(len(nms)))

        print("Loading object masks...")
        imgs, masks, mask_ann_ids = get_segmented_masks([cat], image_folder, coco_data)

        print("Getting number of objects...")
        num_objs = get_num_objs([cat], coco_data)

        print("Getting number of images...")
        num_imgs = get_num_imgs([cat], coco_data)

        print("Getting average number of objects per images...")
        avg_objs_per_img, _ = get_objs_per_img([cat], coco_data)

        print("Getting average area...")
        avg_area, area_data = get_proportion([cat], coco_data, ann_ids=mask_ann_ids)

        print("Getting average roughness of segmentation...")
        avg_roughness, roughness_data = get_seg_roughness(
            [cat], coco_data, ann_ids=mask_ann_ids
        )

        # print("Getting object histograms...")
        # hist_data = getHistograms(imgs, masks)
        hist_data = None

        # print("Getting dominant object colours...")
        # colour_data = getObjColors(imgs, masks)
        colour_data = None

        print("Getting abnormal objects...")
        preds_df = getOutliers(
            hist_data, colour_data, area_data, roughness_data, contamination=0.05
        )
        outlier_img_ids, outlier_ann_ids = getAnomalies(
            [cat], preds_df["lof"], coco_data
        )
        print("Done!")
        print()
        data[len(data)] = [
            cat,
            num_objs,
            num_imgs,
            avg_objs_per_img,
            avg_area,
            avg_roughness,
            outlier_img_ids,
            outlier_ann_ids,
        ]
        cat_num += 1
    df = pd.DataFrame.from_dict(
        data,
        orient="index",
        columns=[
            "category",
            "number of objects",
            "number of images",
            "avg number of objects per img",
            "avg percentage of img",
            "avg num vertices x 1000 / area",
            "images w/ abnormal objects",
            "abnormal objects",
        ],
    )
    print(df)
    timestr = time.strftime("%Y%m%d%H%M%S")
    df.to_pickle("../output/analysis" + timestr + ".pkl")
    return df
