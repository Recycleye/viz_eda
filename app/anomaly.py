import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# from sklearn.decomposition import PCA


def get_outliers(hist, colour, area, roughness, nn=30, contam=0.05):
    """
    :param hist: (hist_size, 3) numpy array of B, G, R histograms
    :param colour: list of dominant colours of objects
    :param area: df containing area of each object, along with its annID
    and imgID
    :param roughness: df containing roughness of each object, along with
    its annID and imgID
    :param nn: number of neighbours used in LOF outlier detection
    :param contam: estimated percentage of outliers/anomalies in the
    given dataset
    :return: df containing annID, lof score (-1 for outlier, 1 for inlier),
    and negative outlier factor of all objects
    """
    train = pd.DataFrame(area["annID"])
    train["area"] = area["proportion of img"]
    train["roughness"] = roughness["roughness"]

    # TODO: Research other colour analysis anomaly detection methods
    # -- histogram method does not work well
    # -- k-means clustering is better, but slow
    # -- anomaly detection with just area and roughness works great

    # histData_2d = []
    # for obj_hist in histData:
    #     # pca = PCA(n_components=5)
    #     # features = pca.fit_transform(obj_hist)
    #     # histData_2d.append(features.flatten())
    #     histData_2d.append(obj_hist.flatten())
    # train = train.join(
    #     pd.DataFrame(histData_2d)
    # )

    # train = train.join(
    #     pd.DataFrame(colourData, columns=["R", "G", "B"])
    # )

    # annIDs should not be included in anomaly detection
    train = train.drop(["annID"], axis=1)
    lof = LocalOutlierFactor(n_neighbors=nn, contamination=contam)
    results = pd.DataFrame()
    results["lof"] = lof.fit_predict(train)
    results["negative_outlier_factor"] = lof.negative_outlier_factor_
    return results


def get_anomalies(cat_ids, preds, coco_data):
    """
    :param cat_ids: list of category IDs
    :param preds: df containing annIDs, lof score
    (-1 for outlier, 1 for inlier), and negative outlier factor of objects
    :param coco_data: loaded coco dataset
    :return: imgIDs of outliers, annIDs of outliers
    """
    img_ids = coco_data.getImgIds(catIds=cat_ids)
    ann_ids = coco_data.getAnnIds(imgIds=img_ids, catIds=cat_ids, iscrowd=0)

    outlier_ann_ids = [id for id, pred in zip(ann_ids, preds) if pred == -1]

    outlier_img_ids = []
    for img in img_ids:
        img_anns = set(coco_data.getAnnIds(imgIds=[img]))
        outlying_anns = set(outlier_ann_ids)
        if len(img_anns.intersection(outlying_anns)) > 0:
            outlier_img_ids.append(img)
    print("Imgs w/ outliers: " + str(outlier_img_ids))
    return outlier_img_ids, outlier_ann_ids
