import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from pycocotools import coco


# def findAnomalies(filterClasses):
#     print("Loading object masks...")
#     segmented_masks = getSegmentedMasks(filterClasses)
#     print("Preparing dataset...")
#     data = loadHistograms(segmented_masks, bins=(3, 3, 3))
#     print("Fitting anomaly detection model...")
#     model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
#     model.fit(data)


def getOutliers(
    histData, areaData, roughnessData, colourData, nn=20, contamination=0.1
):
    colourData_2d = []
    for palette in colourData:
        c = [j for i in palette for j in i]
        colourData_2d.append(c)
    pca = PCA(n_components=3)
    colourData_2d = pca.fit_transform(colourData_2d)
    pca = PCA(n_components=6)
    histData = pca.fit_transform(histData)

    train = pd.DataFrame(areaData["annID"])
    train = train.join(
        pd.DataFrame(
            histData, columns=["hist1", "hist2", "hist3", "hist4", "hist5", "hist6"]
        )
    )
    train["area"] = areaData["proportion of img"]
    train["roughness"] = roughnessData["roughness of annotation"]
    train = train.join(
        pd.DataFrame(colourData_2d, columns=["colourX", "colourY", "colourZ"])
    )
    train = train.drop(["annID"], axis=1)

    print("--Fitting anomaly detection model...")
    lof = LocalOutlierFactor(n_neighbors=nn, contamination=contamination)
    results = pd.DataFrame()
    results["lof"] = lof.fit_predict(train)
    results["negative_outlier_factor"] = lof.negative_outlier_factor_
    return results


def getAnomalies(filterClasses, preds):
    from src.app import cocoData

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
    print("Imgs w/ outliers: " + str(imgs_with_outliers))
    return imgs_with_outliers, outlying_objs_anns
