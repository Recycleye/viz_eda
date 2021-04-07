import json
import os

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from anomaly_analysis.anomaly_feature_extraction import get_HOG
from anomaly_detector import create_destination
from crop_utils import batch_crop_images


def form_crop_image(image_id, annotation_id, cat_id, coco, image_path, crop_destination_path):
    crop_image_filename = f"{image_id}_{annotation_id}_{cat_id}.jpg"
    if not os.path.exists(os.path.join(crop_destination_path, crop_image_filename)):
        batch_crop_images(coco, img_ids=[image_id], img_source=image_path,
                          img_destination=crop_destination_path, proportion=0.05)
    return crop_image_filename


def combine_feature_dataset(annotation_file, img_folder, intermediate_rlt_path, cat_name=[]):
    """
    :param annotation_file: path to JSON coco-style annotation file
    :param imgs_path: path to folder containing images corresponding to
    annotation_file
    :return: final analysis dataframe
    """
    coco_data = COCO(annotation_file)

    for idx, cat in enumerate(cat_name):
        cat = [cat]
        print(cat[0] + ": " + str(idx + 1) + "/" + str(len(cat_name)))

        # get all cat_ids,img_ids,ann_ids in this cat
        cat_ids = coco_data.getCatIds(catNms=cat)
        img_ids = coco_data.getImgIds(catIds=cat_ids)
        ann_ids = coco_data.getAnnIds(catIds=cat_ids)

        print("Getting number of objects...")
        num_objs = len(ann_ids)
        print("the number of objects is:", num_objs)

        print("Getting number of images...")
        num_imgs = len(img_ids)
        print("the number of images is:", num_imgs)

        # add cropped image

        create_destination(intermediate_rlt_path)
        crop_destination_path = os.path.join(intermediate_rlt_path, "crop_bbox_images")
        create_destination(crop_destination_path)

        croped_image = []
        croped_ann_id = []
        croped_image_id = []
        for imgid in img_ids:
            all_ann_ids = coco_data.getAnnIds(imgIds=imgid, catIds=cat_ids, iscrowd=0)
            objs = coco_data.loadAnns(ids=all_ann_ids)
            for obj in objs:
                form_crop_image(image_id=imgid, annotation_id=obj['id'], cat_id=obj['category_id'], coco=coco_data,
                                image_path=img_folder, crop_destination_path=crop_destination_path)
                if os.path.exists(os.path.join(crop_destination_path,
                                               str(obj['image_id']) + "_" + str(obj['id']) + "_" + str(
                                                   obj['category_id']) + ".jpg")):
                    img = cv2.imread(os.path.join(crop_destination_path,
                                                  str(obj['image_id']) + "_" + str(obj['id']) + "_" + str(
                                                      obj['category_id']) + ".jpg"))
                    croped_ann_id.append(obj['id'])
                    croped_image_id.append(obj['image_id'])
                    croped_image.append(img)

        print("form the feature dataset")
        print("Getting HOG extracted data!")
        var, HOG_feature = get_HOG(croped_image, croped_ann_id)
        print("After PCA the explained variance is:", var)
        feature_dataset = HOG_feature

        # ========put ground truth in the feature dataset===========#
        objs = coco_data.loadAnns(ids=croped_ann_id)
        anomaly_label = []
        for obj in objs:
            anomaly_label.append(obj["anomaly"])
        feature_dataset["label"] = anomaly_label

    return feature_dataset, var


def get_outliers(feature_dataset, nn=30, contam=0.05):
    """
    :param feature_dataset: {pd.dataframe}
    :param nn: number of neighbours used in LOF outlier detection
    :param contam: estimated percentage of outliers/anomalies in the
    given dataset
    :return: df containing annID, lof score (-1 for outlier, 1 for inlier),
    and negative outlier factor of all objects
    """

    results = pd.DataFrame()
    train = feature_dataset
    train = train.drop(["annID", "label"], axis=1)
    # ==================== lof 1:inlier -1: outlier============================
    print(train.info())
    print(train)
    if train.shape[0] < nn:
        nn = train.shape[0]
    if train.shape[0] == 1:
        results["lof"] = 1
        results["lof_negative_outlier_factor"] = -1
    else:
        lof = LocalOutlierFactor(n_neighbors=nn, contamination=contam)
        results["lof"] = lof.fit_predict(train)
        # opposite score of lof (1:outlier ,-1 inlier )
        results["lof_negative_outlier_factor"] = lof.negative_outlier_factor_

    # =========== isolation forest 1:inlier -1: outlier========================
    rng = np.random.RandomState(42)
    # feature warning(changing from old version sklearn to new version, you need to specify the behaviour)
    iforest = IsolationForest(n_estimators=100, contamination=contam, random_state=rng, behaviour="new").fit(train)
    results["iforest"] = iforest.fit_predict(train)
    results["iforest_negative_outlier_factor"] = iforest.score_samples(train)
    results["annID"] = feature_dataset["annID"]

    return results


def get_anomalies(predicate, algorithm="lof"):
    """
    :param cat_ids: list of category IDs
    :param preds: df containing annIDs, lof score
    (-1 for outlier, 1 for inlier), and negative outlier factor of objects
    :param coco_data: loaded coco dataset
    :return: imgIDs of outliers, annIDs of outliers
    """
    preds = pd.DataFrame()
    preds["annID"] = predicate["annID"]
    if algorithm == "iforest":
        preds["anomaly"] = predicate["iforest"]
        preds["anomaly_score"] = predicate["iforest_negative_outlier_factor"]
    else:
        preds["anomaly"] = predicate["lof"]
        preds["anomaly_score"] = predicate["lof_negative_outlier_factor"]
    preds = preds[preds["anomaly"] == -1]
    return preds


def detect_anomalies_hog_iforest(annotation_path, images_path, intermediate_rlt_path, cat_ids=None):
    anomaly_path = "output/output_hog_isolationforest.json"
    if os.path.exists(anomaly_path):
        with open(anomaly_path, 'r') as ano_f:
            anomalies = json.load(ano_f)
            return anomalies

    cocoData = COCO(annotation_path)

    # make it apply to all classes
    cats = cocoData.loadCats(cocoData.getCatIds())
    names = [cat["name"] for cat in cats]
    class_result = []

    for idx, cat in enumerate(names):
        cat = [cat]
        print(cat[0] + ": " + str(idx + 1) + "/" + str(len(names)))

        feature_dataset, var = combine_feature_dataset(annotation_file=annotation_path, img_folder=images_path,
                                                       intermediate_rlt_path=intermediate_rlt_path, cat_name=cat)
        print(feature_dataset.info())
        print("After PCA the dataset variance is:", var)

        print("Getting abnormal objects...")
        preds_df = get_outliers(feature_dataset, contam=0.03)

        algorithm = 'iforest'
        anomalies = get_anomalies(preds_df, algorithm)
        print("Outlier_img_id is the following!")
        print(anomalies["annID"])
        print(anomalies["anomaly_score"])
        print("Done!")

        for index, row in anomalies.iterrows():
            anomaly = {
                "id": int(row["annID"]),
                "variance": float(var),
                "anomaly_score": float(row["anomaly_score"])
            }
            class_result.append(anomaly)
    print(class_result)
    with open(anomaly_path, 'w+') as outfile:
        json.dump(class_result, outfile)


def detect_anomalies_hog_lof(annotation_path, images_path, intermediate_rlt_path, cat_ids=None):
    anomaly_path = "output/output_hog_lof.json"
    if os.path.exists(anomaly_path):
        with open(anomaly_path, 'r') as ano_f:
            anomalies = json.load(ano_f)
            return anomalies

    cocoData = COCO(annotation_path)

    # make it apply to all classes
    cats = cocoData.loadCats(cocoData.getCatIds())
    names = [cat["name"] for cat in cats]
    class_result = []

    for idx, cat in enumerate(names):
        cat = [cat]
        print(cat[0] + ": " + str(idx + 1) + "/" + str(len(names)))

        feature_dataset, var = combine_feature_dataset(annotation_file=annotation_path, img_folder=images_path,
                                                       intermediate_rlt_path=intermediate_rlt_path, cat_name=cat)
        print(feature_dataset.info())
        print("After PCA the dataset variance is:", var)

        print("Getting abnormal objects...")
        preds_df = get_outliers(feature_dataset, contam=0.03)

        algorithm = 'lof'
        anomalies = get_anomalies(preds_df, algorithm)
        print("Outlier_img_id is the following!")
        print(anomalies["annID"])
        print(anomalies["anomaly_score"])
        print("Done!")

        for index, row in anomalies.iterrows():
            anomaly = {
                "id": int(row["annID"]),
                "variance": float(var),
                "anomaly_score": float(row["anomaly_score"])
            }
            class_result.append(anomaly)
    print(class_result)
    with open(anomaly_path, 'w+') as outfile:
        json.dump(class_result, outfile)
    return class_result


if __name__ == "__main__":
    annotation_file = "VOC_COCO/annotations/voc_add_anomaly.json"
    image_folder = "VOC_COCO/images"
    intermediate_path = "output/intermediate"
    detect_anomalies_hog_iforest(annotation_file, image_folder, intermediate_path)
