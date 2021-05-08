import glob
import os

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO


def is_large(ann_coco, img_coco, proportion=0.1):
    threshold = float(img_coco['height']) * float(img_coco['width']) * proportion
    return float(ann_coco['bbox'][-1]) * float(ann_coco['bbox'][-2]) >= threshold


def save_bbox_crop(ann_coco, img_coco, img_source, img_destination, proportion=0.1):
    if glob.glob(os.path.join(img_destination, f'{img_coco["id"]}_*')):
        # image already processed
        return
    large_anns = [ann for ann in ann_coco if is_large(ann, img_coco, proportion)]
    im = Image.open(os.path.join(img_source, img_coco['file_name']))
    for ann in large_anns:
        bbox = ann['bbox']
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int((bbox[0] + bbox[2]))
        y2 = int((bbox[1] + bbox[3]))
        # Cropped image of above dimension
        # (It will not change orginal image)
        im_crop = im.crop((x1, y1, x2, y2))
        im_crop.save(os.path.join(img_destination, f"{ann['image_id']}_{ann['id']}_{ann['category_id']}.jpg"), 'JPEG')


# dataDir = 'data'
# dataType = 'val2017'
# annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# cocoData = COCO(annFile)
# catIds = [1]
# imgIds = cocoData.getImgIds(catIds=catIds)
# img_folder = './data/val2017'
# print("Loading object masks...")
#
# loaded_images, masks, color_masks, mask_ann_ids, img_id = get_obj_masks(catIds, [2153], img_folder, cocoData)
# print(img_id)
# print(mask_ann_ids)
# print(len(color_masks))
#
# plt.imshow(color_masks[0])
# plt.show()


def batch_crop_images(coco, img_ids, img_source, img_destination, proportion=0.1):
    imgs = coco.loadImgs(img_ids)
    for i, img in enumerate(imgs):
        if i % 10 == 0 and i > 0:
            print(f"processing: {i}/{len(imgs)} image")
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id']))
        save_bbox_crop(anns, img, img_source, img_destination, proportion)


def view_annotation(coco, ann_id, img_source):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (225, 255, 255)
    # list of annotations
    ann = coco.loadAnns(ann_id)[0]
    # img: coco image object
    img = coco.loadImgs(ann['image_id'])[0]
    im = cv2.imread(os.path.join(img_source, img['file_name']))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # print(im.shape)
    bbox = ann['bbox']
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int((bbox[0] + bbox[2]))
    y2 = int((bbox[1] + bbox[3]))
    # is anomaly, get 1 or -1, if non, get 0
    is_anomaly = ann.get('anomaly', 0)
    catId = ann['category_id']
    cat = coco.loadCats(catId)
    label = cat[0]['name']
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    x3, y3 = x1 + t_size[0] + 3, y1 + t_size[1] + 4
    color = BLUE if is_anomaly == 0 else (RED if is_anomaly == -1 else GREEN)
    cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness=2)
    cv2.rectangle(im, (x1, y1), (x3, y3), color, thickness=-1)
    cv2.putText(im, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    annFile = os.path.join(os.getcwd(), "anomaly_analysis/data/annotations/add_anomaly.json")
    image_path = os.path.join(os.getcwd(), "coco/images")
    crop_destination_path = os.path.join(os.getcwd(), "output/crop_bbox_images")
    coco = COCO(annFile)
    print(coco)

    # run once
    # batch_crop_images(coco, coco.getImgIds(), image_path, crop_destination_path)

    view_annotation(coco, 558342, image_path)
    view_annotation(coco, 70186, image_path)
    view_annotation(coco, 1094662, image_path)
