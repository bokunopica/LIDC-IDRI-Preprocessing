import argparse
import os
import numpy as np
import cv2

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans


def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def ct_img_preprocess(img, middle=None):
    """
    segment_lung function provides tranformation from ct value to normal images, but i dont need the lung segment
    """
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    if middle is None:
        middle = img[100:400, 100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # remove the underflow bins
    img[img == max] = mean
    img[img == min] = mean

    # apply median filter
    img = median_filter(img, size=3)
    # apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img = anisotropic_diffusion(img)
    return img


def segment_lung(img):
    # function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    # mean = np.mean(img)
    # std = np.std(img)
    # img = img - mean
    # img = img / std

    middle = img[100:400, 100:400]
    # mean = np.mean(middle)
    # max = np.max(img)
    # min = np.min(img)
    # # remove the underflow bins
    # img[img == max] = mean
    # img[img == min] = mean

    # # apply median filter
    # img = median_filter(img, size=3)
    # # apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    # img = anisotropic_diffusion(img)
    img = ct_img_preprocess(img, middle)

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    mask = np.ndarray([512, 512], dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask * img


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mask_find_bboxs(mask):
    # 确保 mask 是单通道的
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 将布尔类型转换为 uint8 类型
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255  # 转换为二值图像，True -> 255, False -> 0

    # 确保 mask 是二值图像
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]  # 排除最外层的连通图


def draw_bboxs(image, bboxs):
    for bbox in bboxs:
        x, y, w, h, _ = bbox  # 解包边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制绿色矩形


def convert_bbox_to_yolo(bboxs, img_width, img_height, class_id):
    yolo_bboxes = []
    
    for bbox in bboxs:
        x, y, w, h, _ = bbox  # 解包
        # 计算中心坐标和宽高的比例
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        # 生成 YOLO 格式
        yolo_bbox = [
            class_id,
            x_center,
            y_center,
            width,
            height,
        ]
        yolo_bboxes.append(yolo_bbox)
    
    return yolo_bboxes


def yolo_bbox_to_str(yolo_bbox):
    return f"{yolo_bbox[0]} {round(yolo_bbox[1], 6)} {round(yolo_bbox[2], 6)} {round(yolo_bbox[3], 6)} {round(yolo_bbox[4], 6)}"


def normalize_img(img: np.ndarray) -> np.ndarray:
    img_min = img.min()
    img_max = img.max()

    if img_max - img_min == 0:  # 避免除以零
        return np.zeros_like(img)  # 返回全零数组

    # 归一化到 0-1
    return (img - img_min) / (img_max - img_min)
