import json
import os

import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from tqdm import tqdm
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
    Dense, Reshape, InputLayer, Flatten

from alibi_detect.od import OutlierAE
from crop_utils import batch_crop_images

from anomaly_detector import create_destination

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
    crop_image_filename = f"{image_id}_{annotation_id}_{cat_id}.jpg"
    if not os.path.exists(os.path.join(crop_destination_path, crop_image_filename)):
        batch_crop_images(coco, img_ids=[image_id], img_source=image_path,
                          img_destination=crop_destination_path, proportion=0.05)
    return crop_image_filename

def detect_anomalies_auto_encoder(annotation_path, images_path, intermediate_rlt_path, cat_ids=None):
    anomaly_path = "output/output_auto_encoder.json"
    if os.path.exists(anomaly_path):
        with open(anomaly_path, 'r') as ano_f:
            anomalies = json.load(ano_f)
            return anomalies

    coco_data = COCO(annotation_path)
    evaluation_result = []

    # make it apply to all classes
    cats = coco_data.loadCats(coco_data.getCatIds())
    names = [cat["name"] for cat in cats]

    for idx, cat in enumerate(names):
        cat = [cat]

        # get all cat_ids,img_ids,ann_ids in this cat
        cat_ids = coco_data.getCatIds(catNms=cat)
        img_ids = coco_data.getImgIds(catIds=cat_ids)
        croped_image = []
        croped_ann_id = []
        croped_image_id = []
        create_destination(intermediate_rlt_path)
        crop_destination_path = os.path.join(intermediate_rlt_path, "crop_bbox_images")
        create_destination(crop_destination_path)
        for imgid in img_ids:
            all_ann_ids = coco_data.getAnnIds(imgIds=imgid, catIds=cat_ids, iscrowd=0)
            objs = coco_data.loadAnns(ids=all_ann_ids)

            for obj in objs:
                croped_image_filename = form_crop_image(image_id=imgid, annotation_id=obj['id'],
                                                        cat_id=obj['category_id'],
                                                        coco=coco_data,
                                                        image_path=images_path,
                                                        crop_destination_path=crop_destination_path)

                img = cv2.imread(os.path.join(crop_destination_path,
                                              croped_image_filename))
                croped_ann_id.append(obj['id'])
                croped_image_id.append(obj['image_id'])
                if img is not None:
                    croped_image.append(img)

        temp_result = {
            "croped_ann_id": croped_ann_id,
            "croped_image_id": croped_image_id,
            "cropped_images": croped_image,
            "category": cat[0],
        }
        evaluation_result.append(temp_result)

    class_result = []
    for item in evaluation_result:
        X_train_ls = []
        for img in item["cropped_images"]:
            imgs = np.array(img).astype("float32") / 255
            resized_X = tf.image.resize(tf.ragged.constant(imgs).to_tensor(), [64, 64])
            X_train_ls.append(resized_X)

        X_train = np.array(X_train_ls)

        model_trained = False
        model_path = 'saved_models'  # change to (absolute) directory where model is downloaded
        create_destination(model_path)
        encoding_dim = 4096

        encoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(64, 64, 3)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(encoding_dim, )
            ])

        decoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(encoding_dim,)),
                Dense(4 * 4 * 128),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
            ])

        # initialize outlier detector
        od = OutlierAE(threshold=.019,  # threshold for outlier score
                       encoder_net=encoder_net,  # can also pass AE model instead
                       decoder_net=decoder_net,  # of separate encoder and decoder
                       )
        if not model_trained:
            # train
            od.fit(X_train,
                   epochs=120)

        resized_test = np.reshape(X_train_ls, (len(X_train_ls), 64, 64, 3))
        res = od.predict(resized_test)
        for index, i in tqdm(enumerate(res['data']['is_outlier'])):
            if i == 0:
                anomaly = {
                    "feature_extraction": "Auto-Encoder",
                    "anomaly_score": float(res['data']['instance_score'][index]),
                    "id": int(item['croped_ann_id'][index]),
                }
                class_result.append(anomaly)
    with open(anomaly_path, 'w+') as outfile:
        json.dump(class_result, outfile)

    return class_result


if __name__ == '__main__':
    annotation_file = "VOC_COCO/annotations/voc_add_anomaly.json"
    image_folder = "VOC_COCO/images"
    intermediate_path = "output/intermediate"
    detect_anomalies_auto_encoder(annotation_file, image_folder, intermediate_path)
