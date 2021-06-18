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
from anomaly_detector import create_destination
from crop_utils import batch_crop_images
from anomaly_analysis.anomaly_feature_extraction import get_roughness, get_histograms, \
    get_obj_colors, get_proportion

def form_crop_image(image_id, annotation_id, cat_id, coco, image_path, crop_destination_path):

    """
    :param image_id:{int or str}  image_id of the image
    :param annotation_id:{int or str} annotation id of the image
    :param cat_id:{int or str} category id of image
    :param coco:{coco format} coco dataset
    :param image_path:{str} directory of the image
    :param crop_destination_path:{str} directory to contain the cropped image

    :return:
        crop_image_filename: {str} the filename of the cropped image
    """

    #create the filename
    crop_image_filename = f"{image_id}_{annotation_id}_{cat_id}.jpg"

    #check if the image has already been cropped
    #if not then crop
    if not os.path.exists(os.path.join(crop_destination_path, crop_image_filename)):
        batch_crop_images(coco, img_ids=[image_id], img_source=image_path,
                          img_destination=crop_destination_path, proportion=0.005)
    return crop_image_filename

def combine_feature_dataset(annotation_file, img_folder, intermediate_rlt_path, cat_name=[]):

    """
    :param annotation_file:{str} path to JSON coco-style annotation file
    :param imgs_path:{str} path to folder containing images corresponding to annotation_file
    :param intermediate_rlt_path:{str} path to hold intermediate result
    :param cat_name:{list of str} categories needed to be analyzed

    :return
        feature_dataset:{pd.dataframe} final feature dataframe
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

        num_objs = len(ann_ids)
        print("the number of objects is:", num_objs)

        num_imgs = len(img_ids)
        print("the number of images is:", num_imgs)

        #cropped all images
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

        print("Getting average area...")
        avg_area, area_data = get_proportion(cat_ids, croped_image_id, coco_data, croped_ann_id)

        print("Getting average roughness of segmentation...")
        avg_roughness, roughness = get_roughness(cat_ids, croped_image_id, coco_data, croped_ann_id)

        print("Merge the dataset")
        feature_dataset = area_data.merge(roughness, left_on='annID', right_on='annID')

        assert (feature_dataset["imgID_x"].equals(feature_dataset["imgID_y"]) == True)

        print("Getting colorHist !")
        color_hist = get_histograms(croped_image, croped_ann_id, hist_size=16, hist_range=(0, 256), acc=False)

        feature_dataset = feature_dataset.merge(color_hist, left_on="annID", right_on="annID")

        print("Getting object color vector!")
        obj_color_feature = get_obj_colors(croped_image, croped_ann_id)
        feature_dataset = feature_dataset.merge(obj_color_feature, left_on="annID", right_on="annID")

        #put ground truth in the feature dataset
        objs = coco_data.loadAnns(ids=croped_ann_id)
        anomaly_label = []
        for obj in objs:
            anomaly_label.append(obj["anomaly"])
        feature_dataset["label"] = anomaly_label

    return feature_dataset

def get_outliers(feature_dataset,nn=30, contam=0.05):

    """
    :param feature_dataset: {pd.dataframe} feature dataset generated by combine_feature_dataset
    :param nn:{int} number of neighbours used in LOF outlier detection
    :param contam:{double} estimated percentage of outliers/anomalies in the given dataset

    :return:
        results:{pd.dataframe} df containing annID, lof, isolation forest score (-1 for outlier, 1 for inlier),
    and negative outlier factor of both algorithms all objects
        var :{float} variance of PCA
    """

    #expand the feature
    results = pd.DataFrame()
    train=feature_dataset
    train=train.drop(["annID","imgID_x","imgID_y","label"], axis=1)
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

    #lof algorithm
    n_component=0.8
    if train.shape[0]==1:
        results["lof"] = [1]
        results["iforest"]=[1]
        results["lof_negative_outlier_factor"]=[-100]
        results["iforest_negative_outlier_factor"]=[-1]
        results["annID"] = feature_dataset["annID"]
        return results,1

    # nomalising before pca
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train)

    #perform PCA
    pca = PCA(n_components=n_component)
    final_train = pd.DataFrame(pca.fit_transform(train_s))
    explained_variance = np.sum(pca.explained_variance_ratio_)


    if final_train.shape[0] < nn:
        nn = final_train.shape[0]

    #lof
    lof = LocalOutlierFactor(n_neighbors=nn, contamination=contam)
    results["lof"] = lof.fit_predict(final_train)
    results["lof_negative_outlier_factor"] = lof.negative_outlier_factor_

    # isolation forest algorithm
    rng = np.random.RandomState(42)
    #feature warning(changing from old version sklearn to new version, you need to specify the behaviour)
    iforest = IsolationForest(n_estimators=100, contamination=contam, random_state=rng,behaviour="new").fit(final_train)
    results["iforest"] = iforest.predict(final_train)
    results["iforest_negative_outlier_factor"] = iforest.score_samples(final_train)
    results["annID"]=feature_dataset["annID"]
    return results,explained_variance


def get_anomalies(predicate, algorithm="lof"):
    """
    :param predicate: {pd.dataframe} prediction result generated by get_outlier function
    :param algorithm: {str} classification algorithm to get anomalies

    :return:
        preds: {pd.dataframe} df containing anomalies generated by corresponding algorithm
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

#===== Manually iforest ======
def detect_anomalies_manual_iforest(annotation_path, images_path, intermediate_rlt_path, cat_ids=None):

    """
    :param annotation_file:{str} path to JSON coco-style annotation file
    :param imgs_path:{str} path to folder containing images corresponding to annotation_file
    :param intermediate_rlt_path:{str} path to hold intermediate result
    """

    #generate output json path
    anomaly_path = "output/output_manually_isolationforest.json"
    if os.path.exists(anomaly_path):
        with open(anomaly_path, 'r') as ano_f:
            anomalies = json.load(ano_f)
            return anomalies

    #extract all category names
    cocoData = COCO(annotation_path)
    cats = cocoData.loadCats(cocoData.getCatIds())
    names = [cat["name"] for cat in cats]
    class_result = []

    #begin to analysis
    for idx, cat in enumerate(names):
        cat = [cat]
        print(cat[0] + ": " + str(idx + 1) + "/" + str(len(names)))

        #genertate feature dataset
        feature_dataset = combine_feature_dataset(annotation_file=annotation_path, img_folder=images_path,
                                                  intermediate_rlt_path=intermediate_rlt_path, cat_name=cat)

        #predict anomalies
        print("Getting abnormal objects...")
        preds_df, var = get_outliers(feature_dataset, contam=0.05)

        #get anomalies according to the used algorithm
        algorithm = 'iforest'
        anomalies = get_anomalies(preds_df, algorithm)

        #store result in json
        for index, row in anomalies.iterrows():
            anomaly = {
                "id": int(row["annID"]),
                "variance": var,
                "anomaly_score": float(row["anomaly_score"])
            }
            class_result.append(anomaly)

    with open(anomaly_path, 'w+') as outfile:
        json.dump(class_result, outfile)


def detect_anomalies_manual_lof(annotation_path, images_path, intermediate_rlt_path, cat_ids=None):

    """
    :param annotation_file:{str} path to JSON coco-style annotation file
    :param imgs_path:{str} path to folder containing images corresponding to annotation_file
    :param intermediate_rlt_path:{str} path to hold intermediate result
    """

    #generate output json path
    anomaly_path = "output/output_manually_lof.json"
    if os.path.exists(anomaly_path):
        with open(anomaly_path, 'r') as ano_f:
            anomalies = json.load(ano_f)
            return anomalies

    #extract all category names
    cocoData = COCO(annotation_path)
    cats = cocoData.loadCats(cocoData.getCatIds())
    names = [cat["name"] for cat in cats]
    class_result = []

    for idx, cat in enumerate(names):
        cat = [cat]
        print(cat[0] + ": " + str(idx + 1) + "/" + str(len(names)))

        #genertate feature dataset
        feature_dataset = combine_feature_dataset(annotation_file=annotation_path, img_folder=images_path,
                                                  intermediate_rlt_path=intermediate_rlt_path, cat_name=cat)

        #predict anomalies
        preds_df, var = get_outliers(feature_dataset, contam=0.05)

        #get anomalies according to the used algorithm
        algorithm = 'lof'
        anomalies = get_anomalies(preds_df, algorithm)

        #store result in json
        for index, row in anomalies.iterrows():
            anomaly = {
                "id": int(row["annID"]),
                "variance": var,
                "anomaly_score": float(row["anomaly_score"])
            }
            class_result.append(anomaly)

    with open(anomaly_path, 'w+') as outfile:
        json.dump(class_result, outfile)


if __name__ == "__main__":
    annotation_file = "VOC_COCO/annotations/voc_add_anomaly.json"
    image_folder = "VOC_COCO/images"
    intermediate_path = "output/intermediate"
    detect_anomalies_manual_iforest(annotation_file, image_folder, intermediate_path)
