import numpy as np
import pandas as pd
import cv2
import random
from pycocotools import coco as coco

from anomaly import getOutliers, getAnomalies


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


def getArea(filterClasses):
    # Returns average proportion an object of a given class takes up in the image
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    data = {}
    for img in imgIds:
        imAnn = cocoData.loadImgs(ids=img)[0]
        width = imAnn['width']
        height = imAnn['height']

        annIds = cocoData.getAnnIds(imgIds=img, catIds=catIds)
        objs = cocoData.loadAnns(ids=annIds)

        validObjs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                validObjs.append(obj)
        objs = validObjs
        numObjs = len(objs)

        segAreas = np.zeros(numObjs, dtype=np.float32)
        for ix, obj in enumerate(objs):
            segAreas[ix] = obj['area']

        proportion = round_nearest((sum(segAreas) / len(segAreas)) / (width * height))
        data[len(data)] = [img, proportion]
    df = pd.DataFrame.from_dict(data, orient='index', columns=['imgID', 'proportion of img'])
    avg = df['proportion of img'].sum() / len(df['proportion of img'])
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

    masked_imgs = []
    for img_dict in imgs:
        # Load annotations
        annIds = cocoData.getAnnIds(imgIds=img_dict['id'], catIds=catIds, iscrowd=0)
        anns = cocoData.loadAnns(annIds)
        # Create masked images
        for ann in anns:
            polyVerts = segmentTo2DArray(ann['segmentation'])
            masked_img = maskPixels(polyVerts, img_dict, image_folder)
            masked_imgs.append(masked_img)
    return masked_imgs


def stichImages(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return np.array(cv2.vconcat(im_list_resize))


def getObjColors(image):
    n_colors = 10
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
    # TODO: possible memory allocation error, test on workstation/fix space complexity
    print("--Stitching objects...")
    image = stichImages(segmented_masks)
    print("--Processing dominant colours...")
    counts, palette = getObjColors(image)
    color_patch = displayDominantColors(counts, palette)
    return color_patch


def analyzeDataset(annotation_file, image_folder):
    global cocoData
    cocoData = coco.COCO(annotation_file)
    cats = cocoData.loadCats(cocoData.getCatIds())
    nms = [cat['name'] for cat in cats]

    data = {}
    cat_num = 1
    for cat in nms:
        print(cat + ": " + str(cat_num) + "/" + str(len(nms)))

        print("Getting number of objects...")
        numObjs = getNumObjs([cat])

        print("Getting number of images...")
        numImgs = getNumImgs([cat])

        print("Getting average number of objects per images...")
        avgObjsPerImg, _ = getObjsPerImg([cat])

        print("Getting average area...")
        avgArea, _ = getArea([cat])

        print("Loading object masks...")
        segmented_masks = getSegmentedMasks([cat], image_folder)

        print("Getting dominant colours...")
        colorPatch = getCatColors(segmented_masks)

        print("Getting abnormal objects...")
        preds_df = getOutliers(segmented_masks)
        outlier_imgIds, outlier_annIds = getAnomalies([cat], preds_df['lof'])
        print()
        data[len(data)] = [cat, numObjs, numImgs, avgObjsPerImg, avgArea, colorPatch, outlier_imgIds, outlier_annIds]
        cat_num += 1
    df = pd.DataFrame.from_dict(data, orient='index',
                                columns=['category', 'number of objects', 'number of images',
                                         'avg number of objects per img', 'avg percentage of img', 'dominant colours',
                                         'images w/ abnormal objects', 'abnormal objects'])
    print(df)
    df.to_csv("analysis.csv")
    return df
