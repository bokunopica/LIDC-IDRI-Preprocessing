import streamlit as st
import numpy as np
import os

# 根目录
root_dir = "D:/mycodes/LIDC-IDRI-Preprocessing/data/Image/"

# 获取文件夹列表
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
selected_dir = st.selectbox("Select a Dataset", folders)

# 指定图像文件的目录
image_dir = os.path.join(root_dir, selected_dir)  # 原始图像目录
mask_dir = os.path.join(
    "D:/mycodes/LIDC-IDRI-Preprocessing/data/Mask/", selected_dir
)  # 掩码目录


# 在session_state中初始化image_index
if "image_index" not in st.session_state:
    st.session_state.image_index = 0

if "selected_dir" not in st.session_state:
    st.session_state.selected_dir = ""

if "images" not in st.session_state:
    st.session_state.images = []


# 仅在选中的目录变化时更新session_state
if selected_dir != st.session_state.get("selected_dir"):
    st.session_state.image_index = 0  # 重置索引
    image_index = 0
    st.session_state.selected_dir = selected_dir
    # 获取所有的.npy文件并一次性读取
    image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith(".npy")])
    image_paths = [os.path.join(image_dir, f) for f in image_paths]
    images = []
    mask_images = []
    overlay_images = []
    for path in image_paths:
        image = np.load(os.path.join(image_dir, path))
        mask_image = np.load(
            os.path.join(image_dir, path).replace("Image", "Mask").replace("NI", "MA")
        )
        # # 创建颜色掩码（假设掩码为二值图像）
        colored_mask = np.zeros((*mask_image.shape, 3), dtype=np.uint8)
        colored_mask[mask_image > 0] = [255, 0, 0]  # 红色掩码

        # 创建叠加图像
        alpha = 0.5
        colored_image  = np.stack((image,) * 3, axis=-1)
        overlay = (1 - alpha) * colored_image + alpha * colored_mask
        # overlay[mask_image>0] = [255, 0, 0]
        # # 设置红色掩码的透明度
        # alpha = 128  # 透明度范围 0-255
        # overlay[mask_image > 0] = [255, 0, 0]  # 红色

        images.append(image)
        mask_images.append(colored_mask)
        overlay_images.append(overlay)

    st.session_state.images = images
    st.session_state.mask_images = mask_images
    st.session_state.overlay_images = overlay_images


# 创建滑动条
image_index = st.slider(
    "Select a Nodule Slice", min_value=0, max_value=len(st.session_state.images) - 1
)

# 更新slider的值
st.session_state.image_index = image_index

# 加载当前选择的图像和掩码
selected_image = st.session_state.images[st.session_state.image_index]
selected_mask = st.session_state.mask_images[st.session_state.image_index]
selected_overlay = st.session_state.overlay_images[st.session_state.image_index]


# 使用列显示原始图像、掩码和叠加图像
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Image")
    st.image(selected_image, use_column_width=True, clamp=True, channels="GRAY")

with col2:
    st.subheader("Mask Image")
    st.image(selected_mask, use_column_width=True, clamp=True, channels="RGB")

with col3:
    st.subheader("Overlay Image")
    st.image(selected_overlay, use_column_width=True, clamp=True, channels="RGB")
