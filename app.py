import streamlit as st
import numpy as np
import os

# 根目录
root_dir = 'D:/mycodes/LIDC-IDRI-Preprocessing/data/Image/'

# 获取文件夹列表
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
selected_dir = st.selectbox("Select a Dataset", folders)

# 指定图像文件的目录
image_dir = os.path.join(root_dir, selected_dir)  # 原始图像目录
mask_dir = os.path.join('D:/mycodes/LIDC-IDRI-Preprocessing/data/Mask/', selected_dir)  # 掩码目录

# 获取所有的.npy文件
image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
image_paths = [os.path.join(image_dir, f) for f in image_paths]

# 创建滑动条
image_index = st.slider("Select a Nodule Slice", min_value=0, max_value=len(image_paths) - 1, value=0, key=selected_dir)

# 加载原始图像和掩码
selected_image = image_paths[image_index]
data = np.load(selected_image)

# 加载对应的掩码图像，使用新的命名规则
mask_name = os.path.basename(selected_image).replace('NI', 'MA')  # 假设掩码文件名规则
mask_path = os.path.join(mask_dir, mask_name)
mask_data = np.load(mask_path)

# 创建颜色掩码（假设掩码为二值图像）
colored_mask = np.zeros((*mask_data.shape, 3), dtype=np.uint8)
colored_mask[mask_data > 0] = [255, 0, 0]  # 红色掩码

# 设置透明度
alpha = 0.5

# 创建叠加图像
overlay = (data[..., np.newaxis] * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

# 使用列显示原始图像、掩码和叠加图像
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Image")
    st.image(data, use_column_width=True, clamp=True, channels='GRAY')

with col2:
    st.subheader("Mask Image")
    st.image(mask_data, use_column_width=True, clamp=True, channels='GRAY')

with col3:
    st.subheader("Overlay Image")
    st.image(overlay, use_column_width=True, clamp=True, channels='RGB')
