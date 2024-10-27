import sys
import os
import cv2
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from configparser import ConfigParser
from statistics import median_high

from utils import is_dir_path, segment_lung, mask_find_bboxs, convert_bbox_to_yolo, yolo_bbox_to_str, normalize_img
from pylidc.utils import consensus
from PIL import Image
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action="ignore")

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read("lung.conf")

# Get Directory setting
DICOM_DIR = is_dir_path(parser.get("prepare_dataset", "LIDC_DICOM_PATH"))
MASK_DIR = is_dir_path(parser.get("prepare_dataset", "MASK_PATH"))
IMAGE_DIR = is_dir_path(parser.get("prepare_dataset", "IMAGE_PATH"))
CLEAN_DIR_IMAGE = is_dir_path(parser.get("prepare_dataset", "CLEAN_PATH_IMAGE"))
CLEAN_DIR_MASK = is_dir_path(parser.get("prepare_dataset", "CLEAN_PATH_MASK"))
META_DIR = is_dir_path(parser.get("prepare_dataset", "META_PATH"))

# Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint("prepare_dataset", "Mask_Threshold")

# Hyper Parameter setting for pylidc
confidence_level = parser.getfloat("pylidc", "confidence_level")
padding = parser.getint("pylidc", "padding_size")

META_COLS = [
    "patient_id",
    "nodule_no",
    "slice_no",
    "original_image",
    "mask_image",
    "malignancy",
    "is_cancer",
    "is_clean",
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "slice_thickness",
]


class MakeDataSet:
    def __init__(
        self,
        LIDC_Patients_list,
        IMAGE_DIR,
        MASK_DIR,
        CLEAN_DIR_IMAGE,
        CLEAN_DIR_MASK,
        META_DIR,
        mask_threshold,
        padding,
        confidence_level=0.5,
    ):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding, padding), (padding, padding), (0, 0)]
        self.meta = pd.DataFrame(index=[], columns=META_COLS)

    def calculate_characteristic(self, nodule_annotations, name):
        value_list = []
        for annotation in nodule_annotations:
            value_list.append(getattr(annotation, name))
        return median_high(value_list)

    def calculate_malignancy(self, nodule_annotations):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        malignancy = self.calculate_characteristic(nodule_annotations, "malignancy")
        if malignancy > 3:
            return malignancy, True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, "Ambiguous"

    def save_meta(self, meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(
            meta_list,
            index=META_COLS,
        )
        self.meta = self.meta.append(tmp, ignore_index=True)

    def prepare_dataset(self, seg_lung=True):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        for patient in tqdm(self.IDRI_list):
            pid = patient  # LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()

            nodules = scan.cluster_annotations()
            # TODO annotation 提取医生标注的几个特征
            vol = scan.to_volume()
            print(
                "Patient ID: {} | Dicom Shape: {} | Number of Annotated Nodules: {} | Slice Thickness: {}mm".format(
                    pid,
                    vol.shape,
                    len(nodules),
                    scan.slice_thickness,  # slice_thickness ： z轴扫描步长(即切片厚度)，单位：毫米
                )
            )

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            # nodules: [[Annotation(id=84,scan_id=12), Annotation(id=85,scan_id=12), Annotation(id=86,scan_id=12), Annotation(id=87,scan_id=12)]]
            # len(nodules): 患者被标记的结节数量
            if len(nodules) > 0:
                # Patients with nodules
                for nodule_idx, nodule_annotations in enumerate(nodules):
                    # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                    # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(
                        nodule_annotations, self.c_level, self.padding
                    )
                    lung_np_array = vol[cbbox]
                    # We calculate the malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(
                        nodule_annotations
                    )
                    """
                    其他一些标注的结节特征信息
                    subtlety 细微性
                    internalStructure 内部结构
                    calcification 钙化
                    sphericity 球形度
                    margin 边缘
                    lobulation 分叶状
                    spiculation 毛刺状
                    texture 纹理
                    """
                    subtlety = self.calculate_characteristic(
                        nodule_annotations,
                        "subtlety",
                    )
                    internalStructure = self.calculate_characteristic(
                        nodule_annotations,
                        "internalStructure",
                    )
                    calcification = self.calculate_characteristic(
                        nodule_annotations,
                        "calcification",
                    )
                    sphericity = self.calculate_characteristic(
                        nodule_annotations,
                        "sphericity",
                    )
                    margin = self.calculate_characteristic(
                        nodule_annotations,
                        "margin",
                    )
                    lobulation = self.calculate_characteristic(
                        nodule_annotations,
                        "lobulation",
                    )
                    spiculation = self.calculate_characteristic(
                        nodule_annotations,
                        "spiculation",
                    )
                    texture = self.calculate_characteristic(
                        nodule_annotations,
                        "texture",
                    )

                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        if np.sum(mask[:, :, nodule_slice]) <= self.mask_threshold:
                            continue
                        if seg_lung:
                            # Segment Lung part only
                            lung_segmented_np_array = segment_lung(
                                lung_np_array[:, :, nodule_slice]
                            )
                            # I am not sure why but some values are stored as -0. <- this may result in datatype error in pytorch training # Not sure
                            lung_segmented_np_array[lung_segmented_np_array == -0] = 0
                        else:
                            lung_segmented_np_array = lung_np_array[:,:,nodule_slice]
                        # This itereates through the slices of a single nodule
                        # Naming of each file: NI= Nodule Image, MA= Mask Original
                        nodule_name = "{}_NI{}_slice{}".format(
                            pid[-4:],
                            prefix[nodule_idx],
                            prefix[nodule_slice],
                        )
                        mask_name = "{}_MA{}_slice{}".format(
                            pid[-4:],
                            prefix[nodule_idx],
                            prefix[nodule_slice],
                        )
                        # TODO meta info 其他信息
                        meta_list = [
                            pid[-4:],
                            nodule_idx,
                            prefix[nodule_slice],
                            nodule_name,
                            mask_name,
                            malignancy,
                            cancer_label,
                            False,
                            subtlety,
                            internalStructure,
                            calcification,
                            sphericity,
                            margin,
                            lobulation,
                            spiculation,
                            texture,
                            scan.slice_thickness,
                        ]

                        self.save_meta(meta_list)
                        np.save(
                            patient_image_dir / nodule_name, lung_segmented_np_array
                        )
                        np.save(patient_mask_dir / mask_name, mask[:, :, nodule_slice])
            else:
                print("Clean Dataset", pid)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                # There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                for slice in range(vol.shape[2]):
                    if slice > 50: # 为什么是50？
                        break
                    if seg_lung:
                        lung_segmented_np_array = segment_lung(vol[:, :, slice])
                        lung_segmented_np_array[lung_segmented_np_array == -0] = 0
                        lung_mask = np.zeros_like(lung_segmented_np_array)
                    else:
                        lung_segmented_np_array = vol[:, :, slice]
                        lung_mask = np.zeros_like(lung_segmented_np_array)

                    # CN= CleanNodule, CM = CleanMask
                    # nodule_name = "{}/{}_CN001_slice{}".format(
                    #     pid,
                    #     pid[-4:],
                    #     prefix[slice],
                    # )
                    # mask_name = "{}/{}_CM001_slice{}".format(
                    #     pid,
                    #     pid[-4:],
                    #     prefix[slice],
                    # )
                    nodule_name = "{}_CN001_slice{}".format(
                        pid[-4:],
                        prefix[slice],
                    )
                    mask_name = "{}_CM001_slice{}".format(
                        pid[-4:],
                        prefix[slice],
                    )
                    meta_list = [
                        pid[-4:],
                        slice,
                        prefix[slice],
                        nodule_name,
                        mask_name,
                        0,
                        False,
                        True,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        scan.slice_thickness
                    ]
                    self.save_meta(meta_list)
                    np.save(
                        patient_clean_dir_image / nodule_name, lung_segmented_np_array
                    )
                    np.save(patient_clean_dir_mask / mask_name, lung_mask)
        print("Saved Meta data")
        self.meta.to_csv(self.meta_path + "meta_info.csv", index=False)

    def to_object_detection_dataset(self):
        """
        TODO
        将Images+Mask的图像分割数据集转换为目标检测数据集
        原始结构:
        -- DATA_FOLDER
            -- Clean
                -- Image
                -- Mask
            -- Image
                -- LIDC-IDRI-0001
                    -- 0001_NI000_slice000.npy
                ...
            -- Mask
                -- LIDC-IDRI-0001
                    -- 0001_MA000_slice000.npy
                ...

        输出结构:
        -- DATA_FOLDER
            -- images
                -- train
                    -- 000001.jpg
                    -- 000002.jpg
                -- val
                    -- 000003.jpg
                    -- 000004.jpg
            -- labels
                -- train
                    -- 000001.txt
                    -- 000002.txt
                -- val
                    -- 000003.txt
                    -- 000004.txt
        txt文件:
            45 0.479492 0.688771 0.955609 0.5955
            类别 [x_center, y_center, w, h]

        yml文件: 自己编写
        path: ../path/to # dataset root dir
        train: images/train # train images (relative to 'path') 4 images
        val: images/val # val images (relative to 'path') 4 images
        test: # test images (optional)

        # Classes
        names:
        0: nodule
        """
        # 1. 读取Clean+非Clean的所有folder列表
        clean_folders = [f for f in os.listdir(self.clean_path_img) if f.startswith('LIDC-IDRI')]
        mask_folders = [f for f in os.listdir(self.img_path) if f.startswith('LIDC-IDRI') and f not in clean_folders]
        # 2. 创建保存路径
        train_img_path = 'LIDC-IDRI-COCO/images/train'
        train_txt_path = 'LIDC-IDRI-COCO/lables/train'
        val_img_path = 'LIDC-IDRI-COCO/images/val'
        val_txt_path = 'LIDC-IDRI-COCO/lables/val'
        if not os.path.exists(train_txt_path):
            os.makedirs(train_txt_path)
        if not os.path.exists(train_img_path):
            os.makedirs(train_img_path)
        if not os.path.exists(val_img_path):
            os.makedirs(val_img_path)
        if not os.path.exists(val_txt_path):
            os.makedirs(val_txt_path)
        # 3. 将image和mask的npy文件处理成为image+txt[bboxs]形式并保存到指定路径
        # 3.1 train_test_split
        clean_train, clean_val = train_test_split(clean_folders, random_state=42)
        mask_train, mask_val = train_test_split(mask_folders, random_state=42)
        train_folders = mask_train + clean_train
        val_folders = clean_val + mask_val

        def convert_data(folders, folder_type="train"):
            if folder_type == "train":
                save_img_path = train_img_path
                save_txt_path = train_txt_path
            else:
                save_img_path = val_img_path
                save_txt_path = val_txt_path
            no_data_folder = []
            for folder in tqdm(folders):
                if folder in clean_train:
                    img_dir = os.path.join(self.clean_path_img, folder)
                    is_clean = True
                else:
                    img_dir = os.path.join(self.img_path, folder)
                    is_clean = False

                img_npy_list = os.listdir(img_dir)
                if not img_npy_list:
                    no_data_folder.append(folder)
                    print(f"warning: {folder} has no data;is_clean:{is_clean} no_data_count:{len(no_data_folder)}")
                for img_npy in img_npy_list:
                    img_filename = img_npy.split('.')[0]

                    # 图片处理
                    img = np.load(os.path.join(img_dir, img_npy))
                    normalized_img  = normalize_img(img) # 归一化
                    img_uint8 = (normalized_img * 255).astype(np.uint8)
                    Image.fromarray(img_uint8, mode='L').save(f"{save_img_path}/{img_filename}.jpg") #保存图片

                    # 标注处理 -> bounding box
                    if not is_clean:
                        mask = np.load(
                            os.path.join(img_dir, img_npy).replace("Image", "Mask").replace("NI", "MA")
                        )
                        bboxs = mask_find_bboxs(mask)
                        
                        # 保存标注
                        with open(f"{save_txt_path}/{img_filename}.txt", 'w') as f:
                            bboxs = convert_bbox_to_yolo(bboxs,img.shape[0],img.shape[1],class_id=0)
                            for bbox in bboxs:
                                f.write(yolo_bbox_to_str(bbox))
                                f.write('\n')
                    else:
                        open(f"{save_txt_path}/{img_filename}.txt", 'w').close() # 仅新建文件 无标注
            print(no_data_folder)

        convert_data(train_folders, folder_type="train")
        convert_data(val_folders, folder_type="val")




# 获取mask（灰度图）
mask = cv2.imread(r'\mask.png', cv2.COLOR_BGR2GRAY)
# 转换成二值图
ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)




if __name__ == "__main__":
    # I found out that simply using os.listdir() includes the gitignore file
    # LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list = [f for f in os.listdir(DICOM_DIR) if f.startswith("LIDC-IDRI-")]
    LIDC_IDRI_list.sort()

    test = MakeDataSet(
        LIDC_IDRI_list,
        IMAGE_DIR,
        MASK_DIR,
        CLEAN_DIR_IMAGE,
        CLEAN_DIR_MASK,
        META_DIR,
        mask_threshold,
        padding,
        confidence_level,
    )
    # test.prepare_dataset(seg_lung=False)
    test.to_object_detection_dataset()
