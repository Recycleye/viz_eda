import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


def getOutliers(histData, areaData, roughnessData, nn=20, contamination=0.1):
    histData_2d = []
    for obj_hist in histData:
        pca = PCA(n_components=1)
        features = pca.fit_transform(obj_hist)
        histData_2d.append([features[0][0], features[1][0], features[2][0]])

    train = pd.DataFrame(areaData["annID"])
    train["area"] = areaData["proportion of img"]
    train["roughness"] = roughnessData["roughness of annotation"]
    train = train.join(
        pd.DataFrame(histData_2d, columns=["hist_b", "hist_g", "hist_r"])
    )
    train = train.drop(["annID"], axis=1)

    lof = LocalOutlierFactor(n_neighbors=nn, contamination=contamination)
    results = pd.DataFrame()
    results["lof"] = lof.fit_predict(train)
    results["negative_outlier_factor"] = lof.negative_outlier_factor_
    return results


def getAnomalies(filterClasses, preds, cocoData):
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
