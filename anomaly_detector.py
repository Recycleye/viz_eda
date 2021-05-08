import json
import os

import pandas as pd
from numpy import array
from OUTLIERS import smirnov_grubbs as grubbs
from pycocotools.coco import COCO
from tqdm import tqdm

from crop_utils import batch_crop_images, view_annotation
from dash_app import cache
from imageAI_detector import ImageaiDetector
from label_util import word_distance


class AnomalyDetector():
    def __init__(self, dataset, detector, crop_destination_path, image_path, crop_proportion=0.05):
        self.crop_proportion = crop_proportion
        self.dataset = dataset
        self.detector = detector
        self.crop_destination_path = crop_destination_path
        self.image_path = image_path

    def form_crop_image(self, image_id, annotation_id, cat_id):
        crop_image_filename = f"{image_id}_{annotation_id}_{cat_id}.jpg"
        if not os.path.exists(os.path.join(self.crop_destination_path, crop_image_filename)):
            batch_crop_images(self.dataset, img_ids=[image_id], img_source=self.image_path,
                              img_destination=self.crop_destination_path, proportion=self.crop_proportion)
        return crop_image_filename

    def is_anomaly(self, annotation_id, threshold=0.8):
        cat_id = self.dataset.loadAnns(annotation_id)[0]['category_id']
        label = self.dataset.loadCats(cat_id)[0]['name']
        image_id = self.dataset.loadAnns(annotation_id)[0]['image_id']
        crop_image_filename = self.form_crop_image(image_id, annotation_id, cat_id)
        detections = self.detector.detect_objects(crop_image_filename)
        detections = list(map(lambda dic: {k: v for k, v in dic.items() if k in ['name', 'percentage_probability']},
                              detections))
        if not detections:
            # this annotated object is not large enough, assume not an anomaly
            # print("===no object detected by imageai===")
            return False, []
        for detection in detections:
            wd = word_distance(detection['name'], label)
            if wd > threshold:
                # print(f"===detected, label: {label} detection: {detection['detected_name']}===")
                return False, detections
        print(f"===cannot find label: {label} in image===")
        return True, detections


@cache.memoize()
def detect_anomalies_imageai(analysis_path, image_path, intermediate_rlt_path, cat_ids=None):
    """

    Args:
        cat_ids: list of int: categories that we want to detect
        analysis_path: annotated json file
        image_path: path of images folder
        intermediate_rlt_path: cropped image and output images produced by object detector

    Returns:
        list of anomalies
    """
    anomaly_path = "output/anomaly_imageai.json"
    if os.path.exists(anomaly_path):
        with open(anomaly_path, 'r') as ano_f:
            anomalies = json.load(ano_f)
            return anomalies

    # create object detector
    model_path = './yolo.h5'
    create_destination(intermediate_rlt_path)
    imageai_detection_path = os.path.join(intermediate_rlt_path, "imageai_detection")
    create_destination(imageai_detection_path)
    crop_destination_path = os.path.join(intermediate_rlt_path, "crop_bbox_images")
    create_destination(crop_destination_path)

    imageai_detector = ImageaiDetector(crop_destination_path, imageai_detection_path, model_path)
    # create anomaly detector
    coco = COCO(analysis_path)
    anomaly_detector = AnomalyDetector(coco, imageai_detector, crop_destination_path, image_path, crop_proportion=0.05)

    img_ids = coco.getImgIds()

    # if cat id is not given, get all cat_ids
    cat_ids = coco.getCatIds() if cat_ids is None else cat_ids

    # filter out incorrect cat_ids, i.e. cat_ids not exist
    cat_ids = [id for id in cat_ids if id in set(coco.getCatIds())]

    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=cat_ids)

    anomalies = []
    for ann_id in tqdm(ann_ids, desc='Processing'):
        # for ann_id in tqdm(range(15, 20), desc='Processing'):
        # view_annotation(coco, ann_id, image_path)
        # extract cat name
        cat_id = int(coco.loadAnns(ann_id)[0]['category_id'])
        cat_name = coco.loadCats(cat_id)[0]['name']
        anomaly, detections = anomaly_detector.is_anomaly(ann_id)
        if anomaly:
            anomalies.append({'id': ann_id, 'detections': detections})

    with open(anomaly_path, "w+") as f:
        json.dump(anomalies, f)

    return anomalies


def detect_anomalies_size(analysis_path, image_path=None, intermediate_rlt_path=None, cat_ids=None):
    anomaly_path = "output/anomaly_size.json"
    if os.path.exists(anomaly_path):
        with open(anomaly_path, 'r') as ano_f:
            anomalies = json.load(ano_f)
            return anomalies

    f = open('output/analysis.json', 'r')
    analysis = json.load(f)
    classes = analysis["classes"]
    num_outliers = 0
    anomalies = []
    for cl in tqdm(classes, desc='Processing'):
        print(classes[cl]["name"])
        data = array(classes[cl]["size_avg"]["area"])
        ann_ids = array(classes[cl]["size_avg"]["ann"])
        avg = classes[cl]["size_avg"]["avg"]
        print("Average", avg)
        # perform Grubbs' test and identify index (if any) of the outlier
        o = grubbs.max_test_indices(data, alpha=.01)
        if len(o) > 0:
            print("Outlier size area: ", data[o])
            num_outliers += len(o)
            for ann, size in zip(ann_ids[o], data[o]):
                anomalies.append(
                    {'id': int(ann), 'size': int(size), 'average': int(avg)})

    print("Number of outliers: ", num_outliers)
    print("Proportion: ", num_outliers / int(analysis["total_num_objects"]))

    with open(anomaly_path, "w+") as f:
        json.dump(anomalies, f)

    return anomalies


def test_on_coco():
    annFile = os.path.join(os.getcwd(), "anomaly_analysis/data/annotations/add_anomaly.json")
    crop_destination_path = os.path.join(os.getcwd(), "output/crop_bbox_images")
    output_path = os.path.join(os.getcwd(), "output/imageai_detection")
    image_path = os.path.join(os.getcwd(), "coco/images")
    model_path = os.path.join(os.getcwd(), 'yolo.h5')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    coco = COCO(annFile)
    imageai_detector = ImageaiDetector(crop_destination_path, output_path, model_path)
    anomaly_detector = AnomalyDetector(coco, imageai_detector, crop_destination_path, image_path)

    view_annotation(coco, 587562, image_path)
    print(anomaly_detector.is_anomaly(587562))

    view_annotation(coco, 19751, image_path)
    print(anomaly_detector.is_anomaly(19751))


def create_destination(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def test_on_voc():
    annFile = os.path.join(os.getcwd(), "VOC_COCO/annotations/voc_add_anomaly.json")
    crop_destination_path = os.path.join(os.getcwd(), "VOC_COCO/crop_bbox_images")
    output_path = os.path.join(os.getcwd(), "VOC_COCO/imageai_detection")
    image_path = os.path.join(os.getcwd(), "VOC_COCO/images")
    model_path = os.path.join(os.getcwd(), 'yolo.h5')

    create_destination(crop_destination_path)
    create_destination(output_path)

    coco = COCO(annFile)
    imageai_detector = ImageaiDetector(crop_destination_path, output_path, model_path)
    anomaly_detector = AnomalyDetector(coco, imageai_detector, crop_destination_path, image_path, crop_proportion=0.01)

    # annotation_ids = [1, 2, 3, 14, 60, 221]

    # annotation_ids = [i for i in range(20, 30)]
    annotation_ids = [29]
    for id in annotation_ids:
        print(f"###########processing id:{id}##########")
        view_annotation(coco, id, image_path)
        print(anomaly_detector.is_anomaly(id))


def get_info_from_anns(coco, annotation_id: int):
    annotation_id = int(annotation_id)
    cat_id = int(coco.loadAnns(annotation_id)[0]['category_id'])
    cat_name = coco.loadCats(cat_id)[0]['name']
    ann = coco.loadAnns(annotation_id)[0]
    img = coco.loadImgs(ann['image_id'])[0]
    return {'image_id': ann['image_id'],
            'file_name': img['file_name'],
            'bbox': ann['bbox'],
            'cat_id': cat_id,
            'cat_name': cat_name}


def generate_anomalies(analysis_path: str, detect_anomalies, create_dataframe) -> pd.DataFrame:
    with open(analysis_path, 'r') as ann_f:
        analysis = json.load(ann_f)
        annotation_path = analysis['annotation_path']
        images_path = analysis['images_path']

        intermediate_rlt_path = "./output/intermediate"
        # if 'imageai' in algorithms or len(algorithms) == 0:
        #     # simple logic for now as this is the only working algorithm for now
        anomalies = detect_anomalies(annotation_path, images_path, intermediate_rlt_path,
                                     cat_ids=None)
        coco = COCO(annotation_path)
        return create_dataframe(anomalies, coco, images_path)


def create_dataframe_imageai(anomalies_imageai, coco, images_path):
    df = pd.DataFrame(anomalies_imageai)
    detections = df['detections'].apply(lambda l: max(l, key=lambda dic: float(dic['percentage_probability'])))
    detections_df = pd.DataFrame(list(detections.values))
    df_anomaly = pd.concat([df.drop(['detections'], axis=1), detections_df], axis=1)
    df_anomaly = df_anomaly.rename(columns={"name": "detected_name"})
    df_obj = pd.DataFrame(list(df_anomaly['id'].apply(lambda ann_id: get_info_from_anns(coco, ann_id)).values))
    df_obj['file_name'] = df_obj['file_name'].apply(lambda name: os.path.join(images_path, name))
    df_full = pd.concat([df_anomaly, df_obj], axis=1)
    return df_full


def create_dataframe(anomalies, coco, images_path):
    df = pd.DataFrame(anomalies)
    df_obj = pd.DataFrame(list(df['id'].apply(lambda ann_id: get_info_from_anns(coco, ann_id)).values))
    df_obj['file_name'] = df_obj['file_name'].apply(lambda name: os.path.join(images_path, name))
    df_full = pd.concat([df, df_obj], axis=1)
    return df_full


if __name__ == '__main__':
    # test_on_coco()
    # test_on_voc()

    # view_annotation(COCO("VOC_COCO/annotations/voc_add_anomaly.json"), 87, "VOC_COCO/images")
    # anls = detect_anomalies("VOC_COCO/annotations/voc_add_anomaly.json", "VOC_COCO/images",
    # "voc_intermediate")
    # with open("./voc_intermediate/voc_anomaly.json", "w+") as f:
    # json.dump(anls, f)

    # print(generate_anomalies("").columns)

    generate_anomalies("/Users/Yankang/python_file/viz_eda/output/analysis.json", ["imageai"])
