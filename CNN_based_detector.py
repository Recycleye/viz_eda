import json
import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
from anomaly_detector import create_destination
from crop_utils import batch_crop_images

def get_CNN_feature(cropped_images,cropped_ann_ids):

    """
    :param color_masks:{list} list of cropped image
    :param mask_ann_ids:{list} listof annotation id of the cropped image

    :return:
        explain variance:{double} after using pca the explain variance
        (#of input masked image,50) dataframe of the feature that CNN extracted
    """

    #use ResNet to extract data
    resnet = models.resnet18(pretrained=True)
    # remove last fully-connected layer
    new_classifier = nn.Sequential(*list(resnet.children())[:-1])
    resnet = new_classifier

    # preprocess the data
    # mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_this_class = torch.tensor([])
    for image in tqdm(cropped_images):

        # convert from openCV2 to PIL.
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        img_t = transform(pil_image)
        img_t = torch.unsqueeze(img_t, 0)
        img_this_class = torch.cat((img_this_class, img_t), 0)

    # turn the model in to evaluation mode and convert the result into one dimensional array
    resnet.eval()
    result = resnet(img_this_class)
    result = torch.squeeze(result)
    result_n = result.detach().numpy()

    #nomalising before pca
    scaler = StandardScaler()
    result_n_s=scaler.fit_transform(result_n)

    # apply pca to reduce the dimension
    pca = PCA(n_components=0.8, svd_solver='full')
    final_result=pca.fit_transform(result_n_s)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    data=np.hstack((np.array(cropped_ann_ids).reshape(len(cropped_ann_ids),1),final_result))
    data=pd.DataFrame(data)
    data=data.rename(columns={0: "annID"})

    return explained_variance, data


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
        var:{double} variance of pca retained about the class
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


        print("Getting CNN extracted data")
        var, CNN_feature = get_CNN_feature(croped_image, croped_ann_id)
        feature_dataset = CNN_feature

        #put ground truth in the feature dataset
        objs = coco_data.loadAnns(ids=croped_ann_id)
        anomaly_label = []
        for obj in objs:
            anomaly_label.append(obj["anomaly"])
        feature_dataset["label"] = anomaly_label

    return feature_dataset, var


def get_outliers(feature_dataset, nn=30, contam=0.05):

    """
    :param feature_dataset: {pd.dataframe} feature dataset generated by combine_feature_dataset
    :param nn:{int} number of neighbours used in LOF outlier detection
    :param contam:{double} estimated percentage of outliers/anomalies in the given dataset

    :return:
        results:{pd.dataframe} df containing annID, lof, isolation forest score (-1 for outlier, 1 for inlier),
    and negative outlier factor of both algorithms all objects
    """

    #drop unrelated feature
    results = pd.DataFrame()
    train = feature_dataset
    train = train.drop(["annID", "label"], axis=1)

    if train.shape[0] < nn:
        nn = train.shape[0]

    #if the class only contains 1 object
    if train.shape[0] == 1:
        results["lof"] = [1]
        results["iforest"]=[1]
        results["lof_negative_outlier_factor"]=[-100]
        results["iforest_negative_outlier_factor"]=[-1]
        results["annID"] = feature_dataset["annID"]
        return results

    else:
    #lof algorithm
        lof = LocalOutlierFactor(n_neighbors=nn, contamination=contam)
        results["lof"] = lof.fit_predict(train)

        # opposite score of lof (1:outlier ,-1 inlier )
        results["lof_negative_outlier_factor"] = lof.negative_outlier_factor_

    # isolation forest algorithm
    rng = np.random.RandomState(42)
    # feature warning(changing from old version sklearn to new version, you need to specify the behaviour)
    iforest = IsolationForest(n_estimators=100, contamination=contam, random_state=rng, behaviour="new").fit(train)
    results["iforest"] = iforest.fit_predict(train)
    results["iforest_negative_outlier_factor"] = iforest.score_samples(train)
    results["annID"] = feature_dataset["annID"]
    return results


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


# ===== CNN iforest ======
def detect_anomalies_cnn_iforest(annotation_path, images_path, intermediate_rlt_path, cat_ids=None):

    """
    :param annotation_file:{str} path to JSON coco-style annotation file
    :param imgs_path:{str} path to folder containing images corresponding to annotation_file
    :param intermediate_rlt_path:{str} path to hold intermediate result
    """

    #generate output json path
    anomaly_path = "output/output_CNN_isolationforest.json"
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
        feature_dataset, var = combine_feature_dataset(annotation_file=annotation_path, img_folder=images_path,
                                                       intermediate_rlt_path=intermediate_rlt_path, cat_name=cat)

        #predict anomalies
        preds_df = get_outliers(feature_dataset, contam=0.05)

        #get anomalies according to the used algorithm
        algorithm = 'iforest'
        anomalies = get_anomalies(preds_df, algorithm)

        #store result in json
        for index, row in anomalies.iterrows():
            anomaly = {
                "id": int(row["annID"]),
                "variance": float(var),
                "anomaly_score": float(row["anomaly_score"])
            }
            class_result.append(anomaly)

    with open(anomaly_path, 'w+') as outfile:
        json.dump(class_result, outfile)


# ====== CNN lof =======
def detect_anomalies_cnn_lof(annotation_path, images_path, intermediate_rlt_path, cat_ids=None):

    """
    :param annotation_file:{str} path to JSON coco-style annotation file
    :param imgs_path:{str} path to folder containing images corresponding to annotation_file
    :param intermediate_rlt_path:{str} path to hold intermediate result
    """

    #generate output json path
    anomaly_path = "output/output_CNN_lof.json"
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
        feature_dataset, var = combine_feature_dataset(annotation_file=annotation_path, img_folder=images_path,
                                                       intermediate_rlt_path=intermediate_rlt_path, cat_name=cat)

        #predict anomalies
        preds_df = get_outliers(feature_dataset, contam=0.05)

        #get anomalies according to the used algorithm
        algorithm = 'lof'
        anomalies = get_anomalies(preds_df, algorithm)

        #store result in json
        for index, row in anomalies.iterrows():
            anomaly = {
                "id": int(row["annID"]),
                "variance": float(var),
                "anomaly_score": float(row["anomaly_score"])
            }
            class_result.append(anomaly)
    with open(anomaly_path, 'w+') as outfile:
        json.dump(class_result, outfile)
    return class_result


if __name__ == "__main__":
    annotation_file = "VOC_COCO/annotations/voc_add_anomaly.json"
    image_folder = "VOC_COCO/images"
    intermediate_path = "output/intermediate"
    detect_anomalies_cnn_lof(annotation_file, image_folder, intermediate_path)
