import numpy as np
import pandas as pd
import cv2
import random
from skimage import io
from skimage.color import gray2rgb
from pycocotools import coco as coco
from matplotlib.path import Path


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


def maskPixels(polygon, img):
    path = Path(polygon)
    x, y = np.mgrid[:img['width'], :img['height']]
    points = np.vstack((x.ravel(), y.ravel())).T
    mask = path.contains_points(points)
    img_mask = mask.reshape(x.shape).T
    return img_mask


def getSegmentedMasks(filterClasses):
    # Returns single object annotation with background removed
    dataDir = 'data'
    dataType = 'val2017'
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    if len(imgIds) > 500:
        imgIds = random.sample(imgIds, 500)
    imgs = cocoData.loadImgs(imgIds)

    segmented_masks = []
    for img in imgs:
        # Load image
        loaded_img = io.imread('{}/{}/{}'.format(dataDir, dataType, img['file_name'])) / 255.0
        if len(loaded_img.shape) == 2:
            loaded_img = gray2rgb(loaded_img)

        # Load annotations
        annIds = cocoData.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
        anns = cocoData.loadAnns(annIds)

        for ann in anns:
            polyVerts = segmentTo2DArray(ann['segmentation'])
            img_mask = maskPixels(polyVerts, img)
            segmented_masks.append(loaded_img * img_mask[..., None])
    return segmented_masks


def stichImages(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return np.array(cv2.vconcat(im_list_resize))


def getObjColors(image):
    n_colors = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    pixels = np.float32(image.reshape(-1, 3))
    pixels = pixels[np.all(pixels != 0, axis=1), :]
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    # dominant = palette[np.argmax(counts)]
    palette *= 255.0
    return counts, palette


def displayDominantColors(counts, palette):
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(640*freqs)
    dom_patch = np.zeros(shape=(400, 640, 3), dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
    return dom_patch


def getCatColors(filterClasses):
    print("--Loading object masks...")
    segmented_masks = getSegmentedMasks(filterClasses)
    print("--Stitching objects...")
    image = stichImages(segmented_masks)
    print("--Processing dominant colours...")
    counts, palette = getObjColors(image)
    color_patch = displayDominantColors(counts, palette)
    return color_patch


def analyzeDataset(file):
    # TODO: memory allocation error, test on workstation
    global cocoData
    cocoData = coco.COCO(file)
    # display COCO categories
    cats = cocoData.loadCats(cocoData.getCatIds())
    nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    data = {}
    for cat in nms:
        print(cat)
        print("Getting number of objects...")
        numObjs = getNumObjs([cat])
        print("Getting number of images...")
        numImgs = getNumImgs([cat])
        print("Getting average number of objects per images...")
        avgObjsPerImg, _ = getObjsPerImg([cat])
        print("Getting average area...")
        avgArea, _ = getArea([cat])
        print("Getting dominant colours...")
        colorPatch = getCatColors([cat])
        print("\n")
        data[len(data)] = [cat, numObjs, numImgs, avgObjsPerImg, avgArea, colorPatch]
    df = pd.DataFrame.from_dict(data, orient='index',
                                columns=['category', 'number of objects', 'number of images',
                                         'avg number of objects per img', 'avg percentage of img', 'dominant colours'])
    print(df)
    df.to_pickle("analysis.pkl")
    return df
