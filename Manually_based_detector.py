import json
import json
import os

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from anomaly_analysis.anomaly_feature_extraction import get_roughness, get_histograms, \
    get_obj_colors, get_proportion


def combine_feature_dataset(annotation_file, img_folder, intermediate_rlt_path, cat_name=[]):
    """
    :param annotation_file: path to JSON coco-style annotation file
    :param imgs_path: path to folder containing images corresponding to
    annotation_file
    :return: final analysis dataframe
    """
    coco_data = COCO(annotation_file)

    # detect anomaly in every category
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

        croped_image = []
        croped_ann_id = []
        croped_image_id = []
        for imgid in img_ids:
            all_ann_ids = coco_data.getAnnIds(imgIds=imgid, catIds=cat_ids, iscrowd=0)
            objs = coco_data.loadAnns(ids=all_ann_ids)

            for obj in objs:
                if os.path.exists(os.path.join(intermediate_rlt_path,
                                               str(obj['image_id']) + "_" + str(obj['id']) + "_" + str(
                                                   obj['category_id']) + ".jpg")):
                    img = cv2.imread(os.path.join(intermediate_rlt_path,
                                                  str(obj['image_id']) + "_" + str(obj['id']) + "_" + str(
                                                      obj['category_id']) + ".jpg"))
                    croped_ann_id.append(obj['id'])
                    croped_image_id.append(obj['image_id'])
                    croped_image.append(img)

        print("Getting average area...")
        avg_area, area_data = get_proportion(cat_ids, croped_image_id, coco_data, croped_ann_id)
        print("the average of the area is", avg_area)
        print("the porportion of the area is", area_data)

        # filter_data_proportion = area_data[(area_data['proportion of img'] > 0.005)]
        # print("After filter", filter_data_proportion)
        # new_img_ids = filter_data_proportion["imgID"].values
        # new_annID = filter_data_proportion["annID"].values

        print("Getting average roughness of segmentation...")
        avg_roughness, roughness = get_roughness(cat_ids, croped_image_id, coco_data, croped_ann_id)
        print(roughness)

        print("Merge the dataset")
        feature_dataset = area_data.merge(roughness, left_on='annID', right_on='annID')
        print(feature_dataset)

        assert (feature_dataset["imgID_x"].equals(feature_dataset["imgID_y"]) == True)

        # get object mask
        # img_folder = './data/val2017'
        #
        # loaded_images, masks, color_masks, mask_ann_ids, img_id = get_obj_masks(cat_ids, set(new_img_ids), img_folder,
        #                                                                         coco_data, new_annID)
        # print("The length of loaded_image is: ", len(loaded_images))
        # print("The length of mask is: ", len(color_masks))
        # print("The length of annID is: ", len(mask_ann_ids))

        print("Getting colorHist !")

        color_hist = get_histograms(croped_image, croped_ann_id, hist_size=16, hist_range=(0, 256), acc=False)
        print("The color of this featue is", color_hist.info())

        feature_dataset = feature_dataset.merge(color_hist, left_on="annID", right_on="annID")
        # return a numpy array

        print("Getting object color vector!")

        obj_color_feature = get_obj_colors(croped_image, croped_ann_id)
        feature_dataset = feature_dataset.merge(obj_color_feature, left_on="annID", right_on="annID")
        print(feature_dataset.info())

        # ========put ground truth in the feature dataset===========#
        objs = coco_data.loadAnns(ids=croped_ann_id)
        anomaly_label = []
        for obj in objs:
            anomaly_label.append(obj["anomaly"])

        feature_dataset["label"] = anomaly_label

    return feature_dataset


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

    # some cols contain list
    # so if you use blue,green, red feature ,HOG feature remember to expand it use the following code!

    train = feature_dataset
    train = train.drop(["annID", "imgID_x", "imgID_y", "label"], axis=1)

    red = train['red'].apply(pd.Series)

    # rename each variable is red
    red = red.rename(columns=lambda x: 'red_' + str(x))

    # view the tags dataframe
    train = pd.concat([train[:], red[:]], axis=1)
    train = train.drop(["red"], axis=1)

    blue = train['blue'].apply(pd.Series)

    # rename each variable is red
    blue = blue.rename(columns=lambda x: 'blue_' + str(x))

    # view the tags dataframe
    train = pd.concat([train[:], blue[:]], axis=1)
    train = train.drop(["blue"], axis=1)

    green = train['green'].apply(pd.Series)

    # rename each variable is red
    green = green.rename(columns=lambda x: 'green_' + str(x))

    # view the tags dataframe
    train = pd.concat([train[:], green[:]], axis=1)
    train = train.drop(["green"], axis=1)

    # ==================== lof 1:inlier -1: outlier============================
    print(train.info())
    print(train)

    # perform PCA
    n_component = 0.9

    # nomalising before pca
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train)

    pca = PCA(n_components=n_component)
    final_train = pd.DataFrame(pca.fit_transform(train_s))
    explained_variance = np.sum(pca.explained_variance_ratio_)

    if final_train.shape[0] < nn:
        nn = final_train.shape[0]
    if final_train.shape[0] == 1:
        results["lof"] = 1
        results["lof_negative_outlier_factor"] = -1
    else:
        lof = LocalOutlierFactor(n_neighbors=nn, contamination=contam)
        results["lof"] = lof.fit_predict(final_train)
        # opposite score of lof (1:outlier ,-1 inlier )
        results["lof_negative_outlier_factor"] = lof.negative_outlier_factor_

    # =========== isolation forest 1:inlier -1: outlier========================
    rng = np.random.RandomState(42)

    # feature warning(changing from old version sklearn to new version, you need to specify the behaviour)
    iforest = IsolationForest(n_estimators=100, contamination=contam, random_state=rng, behaviour="new").fit(
        final_train)
    results["iforest"] = iforest.predict(final_train)
    results["iforest_negative_outlier_factor"] = iforest.score_samples(final_train)
    results["annID"] = feature_dataset["annID"]
    return results, explained_variance


def get_anomalies(predicate, coco_data, algorithm="lof"):
    """
    :param cat_ids: list of category IDs
    :param preds: df containing annIDs, lof score
    (-1 for outlier, 1 for inlier), and negative outlier factor of objects
    :param coco_data: loaded coco dataset
    :return: imgIDs of outliers, annIDs of outliers
    """
    #     img_ids = coco_data.getImgIds(catIds=cat_ids)
    #     ann_ids = coco_data.getAnnIds(imgIds=img_ids, catIds=cat_ids, iscrowd=0)
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


def detect_anomalies_manual_iforest(annotation_path, images_path, intermediate_rlt_path, cat_ids=None):
    anomaly_path = "output/output_manually_isolationforest.json"
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
        crop_destination_path = os.path.join(intermediate_rlt_path, "crop_bbox_images")

        feature_dataset = combine_feature_dataset(annotation_file=annotation_path, img_folder=images_path,
                                                  intermediate_rlt_path=crop_destination_path, cat_name=cat)
        print(feature_dataset.info())

        print("Getting abnormal objects...")
        preds_df, var = get_outliers(feature_dataset, contam=0.03)

        algorithm = 'iforest'
        anomalies = get_anomalies(preds_df, algorithm)
        print("Outlier_img_id is the following!")
        print(anomalies["annID"])
        print(anomalies["anomaly_score"])
        print("Done!")

        for index, row in anomalies.iterrows():
            anomaly = {
                "id": int(row["annID"]),
                "variance": var,
                "anomaly_score": float(row["anomaly_score"])
            }
            class_result.append(anomaly)

    print(class_result)
    with open(anomaly_path, 'w+') as outfile:
        json.dump(class_result, outfile)
    return class_result


def detect_anomalies_manual_lof(annotation_path, images_path, intermediate_rlt_path, cat_ids=None):
    anomaly_path = "output/output_manually_lof.json"
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
        crop_destination_path = os.path.join(intermediate_rlt_path, "crop_bbox_images")

        feature_dataset = combine_feature_dataset(annotation_file=annotation_path, img_folder=images_path,
                                                  intermediate_rlt_path=crop_destination_path, cat_name=cat)
        print(feature_dataset.info())

        print("Getting abnormal objects...")
        preds_df, var = get_outliers(feature_dataset, contam=0.03)

        algorithm = 'lof'
        anomalies = get_anomalies(preds_df, algorithm)
        print("Outlier_img_id is the following!")
        print(anomalies["annID"])
        print(anomalies["anomaly_score"])
        print("Done!")

        for index, row in anomalies.iterrows():
            anomaly = {
                "id": int(row["annID"]),
                "variance": var,
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
    detect_anomalies_manual_iforest(annotation_file, image_folder, intermediate_path)
