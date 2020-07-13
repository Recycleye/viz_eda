import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pycocotools import coco

global cocoData
cocoData = coco.COCO('output.json')


def imageHist(image, bins=(4, 6, 3)):
    valid_pix = np.float32(image.reshape(-1, 3))
    valid_pix = valid_pix[np.all(valid_pix != 0, axis=1), :]
    if valid_pix.shape[0] > 0:
        # compute a 3D color histogram over the image and normalize it
        hist = cv2.calcHist(valid_pix, [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        # return the histogram
        return hist


def loadHistograms(images, bins):
    data = []
    for image in images:
        img_float32 = np.float32(image)
        image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)
        features = imageHist(image, bins)
        if features is not None:
            data.append(features)
    return np.array(data)


# def findAnomalies(filterClasses):
#     print("Loading object masks...")
#     segmented_masks = getSegmentedMasks(filterClasses)
#     print("Preparing dataset...")
#     data = loadHistograms(segmented_masks, bins=(3, 3, 3))
#     print("Fitting anomaly detection model...")
#     model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
#     model.fit(data)


def getOutliers(segmented_masks, nn=20, contamination=0.1,):
    print("--Calculating feature histograms...")
    train_features = loadHistograms(segmented_masks, bins=(3, 3, 3))
    lof = LocalOutlierFactor(n_neighbors=nn, contamination=contamination)

    print("--Fitting anomaly detection model...")
    results = pd.DataFrame()
    results['lof'] = lof.fit_predict(train_features)
    results['negative_outlier_factor'] = lof.negative_outlier_factor_
    return results


def getAnomalies(filterClasses, preds):
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)
    annIds = cocoData.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=0)

    outlying_objs_anns = []
    for annId, pred in zip(annIds, preds):
        if pred == -1:
            outlying_objs_anns.append(annId)

    imgs_with_outliers = []
    for img in imgIds:
        img_anns = set(cocoData.getAnnIds(imgIds=[img]))
        outlying_anns = set(outlying_objs_anns)
        if len(img_anns.intersection(outlying_anns)) > 0:
            imgs_with_outliers.append(img)
    return imgs_with_outliers, outlying_objs_anns
