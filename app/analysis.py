import numpy as np
import pandas as pd
import cv2
import random
from pycocotools import coco as coco
from tqdm import tqdm
from anomaly import getOutliers, getAnomalies
import time


def getNumObjs(filterClasses, cocoData):
    # Returns number of objects of a given class
    catIds = cocoData.getCatIds(catNms=filterClasses)
    annIds = cocoData.getAnnIds(catIds=catIds)
    return len(annIds)


def getNumImgs(filterClasses, cocoData):
    # Returns number of imgs with an object of a given class
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)
    return len(imgIds)


def getObjsPerImg(filterClasses, cocoData):
    # Returns average number of objects of a given class in an image
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    data = {}
    for img in imgIds:
        annIds = cocoData.getAnnIds(imgIds=img, catIds=catIds)
        data[len(data)] = [img, len(annIds)]
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["imgID", "number of objs"]
    )
    avg = df["number of objs"].sum() / len(df["number of objs"])
    return avg, df


def getProportion(filterClasses, cocoData, annIds=None):
    # Returns average proportion an object of a given class takes up in the image
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    data = {}
    for imgId in imgIds:
        imAnn = cocoData.loadImgs(ids=imgId)[0]
        all_annIds = cocoData.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=0)
        if annIds is not None:
            all_annIds = list(set(all_annIds).intersection(set(annIds)))
        objs = cocoData.loadAnns(ids=all_annIds)
        for obj in objs:
            if "area" in obj:
                proportion = obj["area"] / (imAnn["width"] * imAnn["height"])
            else:
                polyVerts = segmentTo2DArray(obj["segmentation"])
                area = getArea(polyVerts)
                proportion = area / (imAnn["width"] * imAnn["height"])
            data[len(data)] = [imgId, obj["id"], proportion]
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["imgID", "annID", "proportion of img"]
    )
    avg = df["proportion of img"].sum() / len(df["proportion of img"])
    return avg, df


def getSegRoughness(filterClasses, cocoData, annIds=None):
    # Returns average roughness an object of a given class
    # "Roughness" = number of segmentation vertices / area of obj
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    data = {}
    for imgID in imgIds:
        all_annIds = cocoData.getAnnIds(imgIds=imgID, catIds=catIds, iscrowd=0)
        if annIds is not None:
            all_annIds = list(set(all_annIds).intersection(set(annIds)))
        objs = cocoData.loadAnns(ids=all_annIds)
        for obj in objs:
            num_vertices = len(obj["segmentation"])
            if "area" in obj:
                roughness = num_vertices / obj["area"]
            else:
                polyVerts = segmentTo2DArray(obj["segmentation"])
                area = getArea(polyVerts)
                roughness = (num_vertices * 1000) / area
            data[len(data)] = [imgID, obj["id"], roughness]
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["imgID", "annID", "roughness of annotation"]
    )
    avg = df["roughness of annotation"].sum() / len(df["roughness of annotation"])
    return avg, df


def getArea(polygon):
    Xs = np.array([x[0] for x in polygon])
    Ys = np.array([y[1] for y in polygon])
    return 0.5 * np.abs(np.dot(Xs, np.roll(Ys, 1)) - np.dot(Ys, np.roll(Xs, 1)))


def segmentTo2DArray(segmentation):
    polygon = []
    for partition in segmentation:
        for x, y in zip(partition[::2], partition[1::2]):
            polygon.append((x, y))
    return polygon


def maskPixels(polygon, img_dict, image_folder):
    img = cv2.imread("{}/{}".format(image_folder, img_dict["file_name"]))
    mask = np.zeros(img.shape, dtype=np.uint8)
    polygon = np.int32(polygon)
    mask = cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
    return img, mask


def getSegmentedMasks(filterClasses, image_folder, cocoData):
    # Returns single object annotation with black background
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    if len(imgIds) > 500:
        imgIds = random.sample(imgIds, 500)
    imgs = cocoData.loadImgs(imgIds)

    mask_annIds = []
    masks = []
    loaded_imgs = []
    for img_dict in tqdm(imgs):
        # Load annotations
        annIds = cocoData.getAnnIds(imgIds=img_dict["id"], catIds=catIds, iscrowd=0)
        anns = cocoData.loadAnns(annIds)
        mask_annIds.extend(annIds)
        # Create masked images
        for ann in anns:
            polyVerts = segmentTo2DArray(ann["segmentation"])
            img, mask = maskPixels(polyVerts, img_dict, image_folder)
            masks.append(mask)
            loaded_imgs.append(img)
    return loaded_imgs, masks, mask_annIds


def getHistograms(images, masks):
    data = []
    for image, mask in tqdm(zip(images, masks)):
        img_float32 = np.float32(image)
        bgr_planes = cv2.split(img_float32)
        histSize = 3
        histRange = (0, 256)  # the upper boundary is exclusive
        accumulate = False

        b_hist = cv2.calcHist(
            bgr_planes, [0], mask, [histSize], histRange, accumulate=accumulate
        )
        g_hist = cv2.calcHist(
            bgr_planes, [1], mask, [histSize], histRange, accumulate=accumulate
        )
        r_hist = cv2.calcHist(
            bgr_planes, [2], mask, [histSize], histRange, accumulate=accumulate
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


def getObjColors(images, masks):
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
        domColour = palette[indices[0]]
        data.append(domColour)
    return data


def analyzeDataset(annotation_file, image_folder):
    cocoData = coco.COCO(annotation_file)
    cats = cocoData.loadCats(cocoData.getCatIds())
    nms = [cat["name"] for cat in cats]

    data = {}
    cat_num = 1
    for cat in nms:
        print(cat + ": " + str(cat_num) + "/" + str(len(nms)))

        print("Loading object masks...")
        imgs, masks, mask_annIds = getSegmentedMasks([cat], image_folder, cocoData)

        print("Getting number of objects...")
        numObjs = getNumObjs([cat], cocoData)

        print("Getting number of images...")
        numImgs = getNumImgs([cat], cocoData)

        print("Getting average number of objects per images...")
        avgObjsPerImg, _ = getObjsPerImg([cat], cocoData)

        print("Getting average area...")
        avgArea, areaData = getProportion([cat], cocoData, annIds=mask_annIds)

        print("Getting average roughness of segmentation...")
        avgRoughness, roughnessData = getSegRoughness(
            [cat], cocoData, annIds=mask_annIds
        )

        # print("Getting object histograms...")
        # histData = getHistograms(imgs, masks)
        histData = None

        # print("Getting dominant object colours...")
        # colourData = getObjColors(imgs, masks)
        colourData = None

        print("Getting abnormal objects...")
        preds_df = getOutliers(
            histData, colourData, areaData, roughnessData, contamination=0.05
        )
        outlier_imgIds, outlier_annIds = getAnomalies([cat], preds_df["lof"], cocoData)
        print("Done!")
        print()
        data[len(data)] = [
            cat,
            numObjs,
            numImgs,
            avgObjsPerImg,
            avgArea,
            avgRoughness,
            outlier_imgIds,
            outlier_annIds,
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
