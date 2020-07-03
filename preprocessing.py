import numpy as np
import pandas as pd
import cv2
from skimage import io
from pycocotools import coco as coco
from matplotlib.path import Path


# from app import cocoData


def get_cat_size(filterClasses):
    catIds = cocoData.getCatIds(catNms=filterClasses)
    annIds = cocoData.getAnnIds(catIds=catIds)
    print("Number of objects in the class:", len(annIds))
    return len(annIds)


def get_avg_area(filterClasses):
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    proportionsOfImg = []
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
        proportionsOfImg.append((sum(segAreas) / len(segAreas)) / (width * height))

    return sum(proportionsOfImg) / len(proportionsOfImg)


def analyze_cats(file):
    global cocoData
    cocoData = coco.COCO(file)
    # display COCO categories
    cats = cocoData.loadCats(cocoData.getCatIds())
    nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    data = {}
    for cat in nms:
        catSize = get_cat_size([cat])
        avgArea = get_avg_area([cat])
        data[len(data)] = [cat, catSize, avgArea]
    df = pd.DataFrame.from_dict(data, orient='index', columns=['category', 'size', 'avg percentage of img'])
    print(df)
    return df


def segmentTo2DArray(segmentation):
    polygon = []
    for partition in segmentation:
        for x, y in zip(partition[::2], partition[1::2]):
            polygon.append((x,y))
    return polygon


def maskPixels(polygon, img):
    path = Path(polygon)
    xmin, ymin, xmax, ymax = np.asarray(path.get_extents(), dtype=int).ravel()
    x, y = np.mgrid[:img['width'], :img['height']]
    points = np.vstack((x.ravel(), y.ravel())).T
    mask = path.contains_points(points)
    path_points = points[np.where(mask)]
    img_mask = mask.reshape(x.shape).T
    return img_mask


def getRawImages(filterClasses):
    catIds = coco.getCatIds(catNms=filterClasses)
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(imgIds)
    I_s = []
    for img in imgs:
        I = io.imread('{}/{}/{}'.format(dataDir, dataType, img['file_name'])) / 255.0
        I_s.append(I)
    return I_s


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
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    palette *= 255.0
    return counts, palette


def displayDominantColors(counts, palette):
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(640*freqs)
    dom_patch = np.zeros(shape=(400,640,3), dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
    return dom_patch


def getCatColors(filterClasses):
    print("Loading images...")
    images = getRawImages(filterClasses)
    print("Stitching images...")
    image = stichImages(images)
    print("Processing dominant colours...")
    counts, palette = getObjColors(image)
    color_patch = displayDominantColors(counts, palette)
    return color_patch

