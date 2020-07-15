import numpy as np
import pandas as pd
import cv2
import random
from pycocotools import coco as coco
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from anomaly import getOutliers, getAnomalies
global cocoData
cocoData = coco.COCO('output.json')


def getNumObjs(filterClasses):
    # Returns number of objects of a given class
    catIds = cocoData.getCatIds(catNms=filterClasses)
    annIds = cocoData.getAnnIds(catIds=catIds)
    return len(annIds)


def getNumImgs(filterClasses):
    # Returns number of imgs with an object of a given class
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)
    return len(imgIds)


def getObjsPerImg(filterClasses):
    # Returns average number of objects of a given class in an image
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    data = {}
    for img in imgIds:
        annIds = cocoData.getAnnIds(imgIds=img, catIds=catIds)
        data[len(data)] = [img, len(annIds)]
    df = pd.DataFrame.from_dict(data, orient='index', columns=['imgID', 'number of objs'])
    avg = df['number of objs'].sum() / len(df['number of objs'])
    return avg, df


def round_nearest(x, a=0.05):
    return round(x / a) * a


def getArea(filterClasses, annIds=None):
    # Returns average proportion an object of a given class takes up in the image
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    data = {}
    for imgId in imgIds:
        imAnn = cocoData.loadImgs(ids=imgId)[0]
        all_annIds = cocoData.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=0)
        if annIds is not None:
            all_annIds = list(set(all_annIds).intersection(set(annIds)))
        objs = cocoData.loadAnns(ids=all_annIds)
        for obj in objs:
            proportion = obj['area'] / (imAnn['width'] * imAnn['height'])
            proportion = round_nearest(proportion)
            data[len(data)] = [imgId, obj['id'], proportion]
    df = pd.DataFrame.from_dict(data, orient='index', columns=['imgID', 'annID', 'proportion of img'])
    avg = df['proportion of img'].sum() / len(df['proportion of img'])
    return avg, df


def getSegRoughness(filterClasses, annIds=None):
    # Returns average roughness an object of a given class
    # "Roughness" = number of segmentation vertices / area of obj
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    data = {}
    for imgID in imgIds:
        all_annIds = cocoData.getAnnIds(imgIds=imgID, catIds=catIds, iscrowd=0)
        if annIds is not None:
            all_annIds = list(set(all_annIds).intersection(set(annIds)))
        objs = cocoData.loadAnns(ids=all_annIds)
        for obj in objs:
            num_vertices = len(obj['segmentation'])
            roughness = num_vertices / obj['area']
            data[len(data)] = [imgID, obj['id'], roughness]
    df = pd.DataFrame.from_dict(data, orient='index', columns=['imgID', 'annID', 'roughness of annotation'])
    avg = df['roughness of annotation'].sum() / len(df['roughness of annotation'])
    return avg, df


def segmentTo2DArray(segmentation):
    polygon = []
    for partition in segmentation:
        for x, y in zip(partition[::2], partition[1::2]):
            polygon.append((x, y))
    return polygon


def maskPixels(polygon, img_dict, image_folder):
    img = cv2.imread('{}/{}'.format(image_folder, img_dict['file_name']))
    mask = np.zeros(img.shape, dtype=np.uint8)
    polygon = np.int32(polygon)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    masked_image = cv2.bitwise_and(img, mask)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    return masked_image


def getSegmentedMasks(filterClasses, image_folder):
    # Returns single object annotation with black background
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    if len(imgIds) > 500:
        imgIds = random.sample(imgIds, 500)
    imgs = cocoData.loadImgs(imgIds)

    mask_annIds = []
    masked_imgs = []
    for img_dict in tqdm(imgs):
        # Load annotations
        annIds = cocoData.getAnnIds(imgIds=img_dict['id'], catIds=catIds, iscrowd=0)
        anns = cocoData.loadAnns(annIds)
        mask_annIds.extend(annIds)
        # Create masked images
        for ann in anns:
            polyVerts = segmentTo2DArray(ann['segmentation'])
            masked_img = maskPixels(polyVerts, img_dict, image_folder)
            valid_pix = np.float32(masked_img.reshape(-1, 3))
            valid_pix = valid_pix[np.all(valid_pix != 0, axis=1), :]
            if valid_pix.shape[0] > 0:
                masked_imgs.append(masked_img)
    return masked_imgs, mask_annIds


def stichImages(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return np.array(cv2.vconcat(im_list_resize))


def getObjColors(image):
    n_colors = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    pixels = np.float32(image.reshape(-1, 3))
    pixels = pixels[np.all(pixels != 0, axis=1), :]
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    return counts, palette


def displayDominantColors(counts, palette):
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(640*freqs)
    dom_patch = np.zeros(shape=(400, 640, 3), dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
    return dom_patch


def getCatColors(segmented_masks):
    print("--Processing object colours")
    colourData = []
    for mask in tqdm(segmented_masks):
        _, palette = getObjColors(mask)
        colourData.append(palette)
    print("--Stitching objects...")
    image = stichImages(segmented_masks)
    print("--Processing category colours...")
    counts, palette = getObjColors(image)
    # color_patch = displayDominantColors(counts, palette)
    return palette, colourData


def imageHist(image, bins=(4, 6, 3)):
    # compute a 3D color histogram over the image and normalize it
    hist = cv2.calcHist(image, [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def getHistograms(images, bins):
    data = []
    for image in tqdm(images):
        img_float32 = np.float32(image)
        image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)
        features = imageHist(image, bins)
        data.append(features)
    return np.array(data)


def analyzeDataset(annotation_file, image_folder):
    global cocoData
    cocoData = coco.COCO(annotation_file)
    cats = cocoData.loadCats(cocoData.getCatIds())
    nms = [cat['name'] for cat in cats]

    data = {}
    cat_num = 1
    for cat in nms:
        print(cat + ": " + str(cat_num) + "/" + str(len(nms)))

        print("Loading object masks...")
        segmented_masks, mask_annIds = getSegmentedMasks([cat], image_folder)

        print("Getting number of objects...")
        numObjs = getNumObjs([cat])

        print("Getting number of images...")
        numImgs = getNumImgs([cat])

        print("Getting average number of objects per images...")
        avgObjsPerImg, _ = getObjsPerImg([cat])

        print("Getting average area...")
        avgArea, areaData = getArea([cat], annIds=mask_annIds)

        print("Getting average roughness of segmentation...")
        avgRoughness, roughnessData = getSegRoughness([cat], annIds=mask_annIds)

        print("Getting dominant colours...")
        catColours, colourData = getCatColors(segmented_masks)

        print("Getting object histograms...")
        histData = getHistograms(segmented_masks, bins=(3, 3, 3))

        print("Getting abnormal objects...")
        preds_df = getOutliers(histData, areaData, roughnessData, colourData, contamination=0.01)
        outlier_imgIds, outlier_annIds = getAnomalies([cat], preds_df['lof'])
        print("Done!")
        print()
        data[len(data)] = [cat, numObjs, numImgs, avgObjsPerImg, avgArea, avgRoughness,
                           catColours, outlier_imgIds, outlier_annIds]
        cat_num += 1
    df = pd.DataFrame.from_dict(data, orient='index',
                                columns=['category', 'number of objects', 'number of images',
                                         'avg number of objects per img', 'avg percentage of img',
                                         'avg num vertices / area', 'dominant colours', 'images w/ abnormal objects',
                                         'abnormal objects'])
    print(df)
    df.to_pickle("analysis.pkl")
    return df
