import os
import pandas as pd
import shutil
import random

# 修改字母即可

# 定义路径
train_folder = 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Train_All'  # 图像所在文件夹
label_file = 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Training_Tag.xlsx'  # 标签文件路径
output_folder_1 = 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Train_D/D'  # 标签为1的图像文件夹
output_folder_2 = 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Train_D/Normal'  # 标签为0的图像文件夹

# 创建输出文件夹，如果没有的话
os.makedirs(output_folder_1, exist_ok=True)
os.makedirs(output_folder_2, exist_ok=True)

# 读取标签文件
df = pd.read_excel(label_file)

# 获取标签为1的图像文件名
label_1_images_L = df[df['D'] == 1]['Left-Fundus'].tolist()
label_1_images_R = df[df['D'] == 1]['Right-Fundus'].tolist()

# 获取标签为0的图像对（ID相同，左右眼对应）
# 首先提取所有正常样本的ID
normal_ids = df[df['N'] == 1]['ID'].unique().tolist()
# 创建一个字典来存储ID对应的左右眼图像
id_to_images = {}
for id in normal_ids:
    left_images = df[(df['ID'] == id) & (df['N'] == 1)]['Left-Fundus'].tolist()
    right_images = df[(df['ID'] == id) & (df['N'] == 1)]['Right-Fundus'].tolist()
    id_to_images[id] = {
        'left': left_images[0] if left_images else None,
        'right': right_images[0] if right_images else None
    }

# 随机选择和标签1数量一样多的ID（确保左右眼对应）
num_label_1 = len(label_1_images_L) + len(label_1_images_R)
# 我们需要大约num_label_1/2个ID（因为每个ID平均有2个图像）
num_ids_needed = max(1, round(num_label_1 / 2))
random_ids = random.sample(normal_ids, min(num_ids_needed, len(normal_ids)))
# 收集这些ID对应的所有图像
random_label_0_images = []
for id in random_ids:
    if id_to_images[id]['left']:
        random_label_0_images.append(id_to_images[id]['left'])
    if id_to_images[id]['right']:
        random_label_0_images.append(id_to_images[id]['right'])

# 移动标签为1的图像到output_1文件夹
for image in label_1_images_L:
    image_path = os.path.join(train_folder, image)
    shutil.copy(image_path, os.path.join(output_folder_1, image))
for image in label_1_images_R:
    image_path = os.path.join(train_folder, image)
    shutil.copy(image_path, os.path.join(output_folder_1, image))

# 移动随机选择的标签为0的图像到output_2文件夹
for image in random_label_0_images:
    image_path = os.path.join(train_folder, image)
    shutil.copy(image_path, os.path.join(output_folder_2, image))
