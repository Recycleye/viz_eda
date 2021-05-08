import random

from pycocotools.coco import COCO
from crop_utils import batch_crop_images, view_annotation
from glob import glob
from imageai.Detection import ObjectDetection
import os

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class ImageaiDetector(object):
    def __init__(self, input_file, output_file, model_path, detection_speed="fast"):
        self.input_file = input_file
        self.output_file = output_file
        self.model_path = model_path

        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath(self.model_path)
        self.detector.loadModel(detection_speed=detection_speed)

    def detect_objects(self, input_image, output_image=""):
        input_image_path = os.path.join(self.input_file, input_image)
        if not os.path.exists(input_image_path):
            return []
        return self.detector.detectObjectsFromImage(input_image=input_image_path,
                                                    output_image_path=os.path.join(self.output_file, output_image
                                                    if output_image else input_image),
                                                    minimum_percentage_probability=90)


if __name__ == '__main__':
    annFile = os.path.join(os.getcwd(), "anomaly_analysis/data/annotations/add_anomaly.json")
    crop_destination_path = os.path.join(os.getcwd(), "output/crop_bbox_images")
    output_path = os.path.join(os.getcwd(), "output/imageai_detection")
    image_path = os.path.join(os.getcwd(), "coco/images")
    model_path = os.path.join(os.getcwd(), 'yolo.h5')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    coco = COCO(annFile)

    # image_names = ["368212_1096288_70", "58539_561350_1", "413552_2203727_1", "439715_55338_19"]
    # image_names = ["368212_1096288_70"]
    # img_files = [f"{name}.jpg" for name in image_names]
    img_files = list(map(os.path.basename, glob(os.path.join(crop_destination_path, "*.jpg"))))

    imageai_detector = ImageaiDetector(crop_destination_path, output_path, model_path)
    for img_file in random.sample(img_files, 10):
        # for img_file in img_files:
        print("======", img_file, "======")
        detections = imageai_detector.detect_objects(img_file)
        for eachObject in detections:
            print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        view_annotation(coco, int(img_file.split("_")[-2]), image_path)
        print("--------------------------------")
