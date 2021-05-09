import json
from skimage.transform import resize
from skimage.feature import hog
import pandas as pd
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import cv2
import os
from PIL import Image
from torch import nn
from sklearn.decomposition import PCA
from torchvision import models
import torch
from sklearn.preprocessing import StandardScaler


def get_features(analysis_path: str) -> pd.DataFrame:
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
        images = analysis["images"]
        empty_img = analysis["empty_images"]

        df_img = pd.DataFrame.from_dict(images, orient="index")
        df_empty_img = pd.DataFrame(empty_img)
        df_empty_img = df_empty_img.rename(columns={0: 'empty_image_id'})
        df_img["image_id"] = df_img.index.map(int)

        # remove rows of empty image
        filtered_df = df_img[~df_img['image_id'].isin(df_empty_img['empty_image_id'])]

        objs_partial = filtered_df['objects']

        # drop column 'classes'
        filtered_df = filtered_df.drop(['classes'], axis=1)

        # select metrics for each object in each image and stack them up
        objs_partial = objs_partial.apply(lambda objs: [{"category_id": obj["category_id"],
                                                         "id": obj["id"],
                                                         "segmentation": obj["segmentation"],
                                                         "area": obj["area"],
                                                         "roughness": len(obj["segmentation"]) / obj["area"],
                                                         "bbox": obj["bbox"],
                                                         "image_id": obj["image_id"]} for obj in objs])

        obj_df = pd.concat([pd.DataFrame(obj) for obj in objs_partial])

        df_joined = obj_df.merge(filtered_df, on="image_id", how='left')

        df_joined['index'] = range(1, len(df_joined) + 1)

        # Class id name dict
        classes = analysis['classes']
        class_id_name = {int(class_id): classes[class_id]['name'] for class_id in classes}
        df_joined['category'] = df_joined['category_id'].apply(class_id_name.get)

        # Rearrange col
        cols = ['index', 'category_id', 'category', 'id', 'segmentation', 'area', 'roughness', 'image_id', 'bbox',
                'width', 'height',
                'file_name']

        # sort by category id / class
        df_joined = df_joined.sort_values(by=['category_id'])

        return df_joined[cols]


def get_proportion(cats, img_ids, coco_data, ann_ids=None):
    """
    :param cats: list of category ids
    :param img_ids: list of image ids in specified categories
    :param coco_data: loaded coco dataset
    :param ann_ids: list of object IDs
    :return: average proportion an object of the given classes take up in the
    image, uses all objs if ann_ids is None
    :return: df containing area of each object, along with its annID and imgID
    """
    data = []
    for imgid in set(img_ids):
        im_ann = coco_data.loadImgs(ids=imgid)[0]
        all_ann_ids = coco_data.getAnnIds(imgIds=imgid, catIds=cats, iscrowd=0)
        if ann_ids is not None:
            all_ann_ids = list(set(all_ann_ids).intersection(set(ann_ids)))

        objs = coco_data.loadAnns(ids=all_ann_ids)
        for obj in objs:

            # Check if annotation file includes precomputed area for object
            # If not, area needs to be calculated
            if "area" in obj:
                proportion = obj["area"] / (im_ann["width"] * im_ann["height"])
            else:
                poly_verts = segment_to_2d_array(obj["segmentation"])
                area = get_area(poly_verts)
                proportion = area / (im_ann["width"] * im_ann["height"])
            data.append((imgid, obj["id"], proportion))
    df = pd.DataFrame(data, columns=["imgID", "annID", "proportion of img"])
    avg = df["proportion of img"].sum() / len(df["proportion of img"])
    return avg, df


def get_area(polygon):
    """
    :param polygon: list of (x, y) points defining a polygon
    :return: area of the polygon
    """
    xs = np.array([x[0] for x in polygon])
    ys = np.array([y[1] for y in polygon])
    v = np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    return 0.5 * v


def get_roughness(cats, img_ids, coco_data, ann_ids=None):
    """
    :param cats: list of category ids
    :param img_ids: list of image ids in specified categories
    :param coco_data: loaded coco dataset
    :param ann_ids: list of object IDs
    :return: average roughness an object of the given classes, uses all objs
    if ann_ids is None
    :return: df containing roughness of each object, along with its annID+imgID
             "roughness" = number of segmentation vertices / area of obj
    """
    data = []
    for imgID in set(img_ids):

        all_ann_ids = coco_data.getAnnIds(imgIds=imgID, catIds=cats, iscrowd=0)
        if ann_ids is not None:
            all_ann_ids = list(set(all_ann_ids).intersection(set(ann_ids)))

        objs = coco_data.loadAnns(ids=all_ann_ids)
        for obj in objs:

            num_vertices = len(obj["segmentation"])

            # Check if annotation file includes precomputed area for object
            # If not, area needs to be calculated
            if "area" in obj:
                roughness = num_vertices / obj["area"]
            else:
                poly_verts = segment_to_2d_array(obj["segmentation"])
                area = get_area(poly_verts)
                roughness = num_vertices / area
            data.append((imgID, obj["id"], roughness * 1000))
    df = pd.DataFrame(data, columns=["imgID", "annID", "roughness"])
    avg = df["roughness"].sum() / len(df["roughness"])
    return avg, df


def segment_to_2d_array(segmentation):
    """
    :param segmentation: a list of segmentation coordinates from coco-style dataset
    :return: list of (x,y) defining a ploygon
    """
    polygon = []
    for partition in segmentation:
        # (0,1),(2,3)
        for x, y in zip(partition[::2], partition[1::2]):
            polygon.append((x, y))
    return polygon


def mask_pixels(polygon, img_dict, image_folder):
    """
    :param polygon: list of (x, y) points defining a polygon
    :param img_dict: image metadata from JSON annotation data
    :param image_folder: path to folder containing images
    :return: loaded image and mask of object
    """
    # print(f"img_folder:{image_folder}")
    # print(f"img_dict:{img_dict}")

    img = cv2.imread(os.path.join(image_folder, img_dict["file_name"]))

    # fill background with black
    mask = np.zeros(img.shape, dtype=np.uint8)
    polygon = pd.np.int32(polygon)
    # fill object with white
    mask = cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
    color_mask = cv2.bitwise_and(img, mask)
    assert (np.array(img).shape == np.array(img).shape)
    return img, mask, color_mask


def get_obj_masks(cats, img_ids, image_folder, coco_data, annsID=None, sample=3):
    """
    :param cats: list of category ids
    :param img_ids: list of image ids in specified categories
    :param image_folder: path to folder containing images
    :param coco_data: loaded coco dataset
    :param sample: sample number of imgs to speed up processing
    :return: loaded images, list of object masks, and objIDs specified class
    """
    imgs = coco_data.loadImgs(img_ids)
    mask_ann_ids = []
    masks = []
    loaded_imgs = []
    color_masks = []
    img_id = []

    for img_dict in tqdm(imgs):
        # Load annotations
        id = img_dict["id"]
        ann_ids = coco_data.getAnnIds(imgIds=id, catIds=cats, iscrowd=0)
        if annsID is not None:
            ann_ids = list(set(ann_ids).intersection(set(annsID)))
        anns = coco_data.loadAnns(ann_ids)
        mask_ann_ids.extend(ann_ids)

        # Create masked images
        for ann in anns:
            poly_verts = segment_to_2d_array(ann["segmentation"])
            img, mask, color_mask = mask_pixels(poly_verts, img_dict, image_folder)
            # append mask and original image to different arrays
            masks.append(mask)
            loaded_imgs.append(img)
            color_masks.append(color_mask)
            img_id.append(img_dict["id"])
            assert (img.shape == np.array(mask).shape)

    return loaded_imgs, masks, color_masks, mask_ann_ids, img_id


def get_HOG(color_masks, mask_ann_ids):
    """
    :param color_masks: list of cropped image
    :param mask_ann_ids: list of annotation id of the image

    :return: average roughness an object of the given classes, uses all objs
    if ann_ids is None
    :return: df containing the hog feature vector of the color_masks

    """
    HOG_feature = []

    for i in tqdm(color_masks):
        # resize the image
        resized_img = resize(i, (128, 64))

        # use hog detector to capture the whole image 3780 dimensions
        fd = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True)
        HOG_feature.append(fd)


    n_component = 0.8
    HOG_feature = pd.DataFrame(HOG_feature)

    # nomalising before pca
    scaler = StandardScaler()
    HOG_feature_n_s = scaler.fit_transform(HOG_feature)

    pca = PCA(n_components=n_component)
    final_result = pd.DataFrame(pca.fit_transform(HOG_feature_n_s))
    explained_variance = np.sum(pca.explained_variance_ratio_)
    final_result["annID"] = mask_ann_ids
    return explained_variance, final_result


def get_histograms(color_masks, annIDs, hist_size=50, hist_range=(0, 256), acc=False):
    """
    :param images: list of loaded images
    :param hist_size: number of bins in histograms
    :param hist_range: range of values to include in histograms
    :param acc: flag to clear histogram before calculation
    :return: (hist_size, 3) numpy array of B, G, R histograms
    """
    data = []
    for image in tqdm(color_masks):
        img_float32 = np.float32(image)
        image_split = cv2.split(img_float32)

        b_hist = cv2.calcHist(
            image_split, [0], None, [hist_size], hist_range, accumulate=acc)

        g_hist = cv2.calcHist(
            image_split, [1], None, [hist_size], hist_range, accumulate=acc)

        r_hist = cv2.calcHist(
            image_split, [2], None, [hist_size], hist_range, accumulate=acc)

        hist_h = 10

        # Notice that before drawing, we first cv::normalize the histogram
        # so its values fall in the range indicated by the parameters entered:

        b_features = cv2.normalize(
            b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX).flatten()

        g_features = cv2.normalize(
            g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX).flatten()
        r_features = cv2.normalize(
            r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX).flatten()

        data.append([b_features, g_features, r_features])

    color_hist = pd.DataFrame(data, columns=["blue", "green", "red"])
    color_hist["annID"] = annIDs

    return color_hist


def get_CNN_feature(color_masks,mask_ann_ids):
    """
    :param color_masks: list of masked image
    :param mask_ann_ids: the annotation id of the masked image

    :return:
        explain variance: after using pca the explain vaiance
        (#of input masked image,50) dataframe of the feature that CNN extracted


    """

    resnet = models.resnet18(pretrained=True)

    # remove last fully-connected layer
    new_classifier = nn.Sequential(*list(resnet.children())[:-1])
    resnet = new_classifier

    # preprocess the data
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    transform = transforms.Compose([  # [1]
        transforms.Resize([256, 256]),  # [2]
        transforms.ToTensor(),
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )
    ])
    img_this_class = torch.tensor([])

    for image in tqdm(color_masks):
        #print(image.type)

        # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that
        # the color is converted from BGR to RGB
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        img_t = transform(pil_image)
        img_t = torch.unsqueeze(img_t, 0)
        img_this_class = torch.cat((img_this_class, img_t), 0)
    # turn the model in to evaluation mode
    resnet.eval()
    result = resnet(img_this_class)
    result = torch.squeeze(result)

    # apply pca to reduce the dimension
    result_n = result.detach().numpy()

    #nomalising before pca
    scaler = StandardScaler()
    result_n_s=scaler.fit_transform(result_n)

    pca = PCA(n_components=0.8, svd_solver='full')
    final_result=pca.fit_transform(result_n_s)
    explained_variance = np.sum(pca.explained_variance_ratio_)

    data=np.hstack((np.array(mask_ann_ids).reshape(len(mask_ann_ids),1),final_result))
    data=pd.DataFrame(data)
    data=data.rename(columns={0: "annID"})

    return explained_variance, data


def get_obj_colors(color_masks, annIDs, n_colors=4):
    """
    :param color_masks: list of cropped image
    :return: list of dominant colours of each object specified by its mask
    """
    # k cluster termination: whenevern 20 iteratios or accuracy of epsilon=0.1 is reached, stop
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    data = []
    for color_mask in tqdm(color_masks):
        # convert to rbg from bgr
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)

        # reshape into a list of pixels: unknow rows, 3 columns
        pixels = np.float32(color_mask.reshape(-1, 3))

        # get non-black pixels
        pixels = pixels[np.all(pixels != 0, axis=1), :]

        # labels: label array where each element marked '0','1',..
        # centers: array of centers of clusters, one row per each cluster center.
        _, labels, centers = cv2.kmeans(pixels, K=n_colors, bestLabels=None, criteria=criteria, attempts=10,
                                        flags=flags)

        # get counts of each unique label
        _, counts = np.unique(labels, return_counts=True)

        # sort counts in descending order, return indices
        indices = np.argsort(counts)[::-1]
        dom_colour = centers[indices[0]]
        data.append(dom_colour)

    obj_color = pd.DataFrame(data, columns=['dom_r', 'dom_b', 'dom_g'])
    obj_color['annID'] = annIDs

    return obj_color
