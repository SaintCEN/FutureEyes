import os
import pandas as pd
import shutil
import random

# 定义路径
train_folder = 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Train_All'
label_file = 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Training_Tag.xlsx'
output_folder_1 = 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Train_D/D'
output_folder_2 = 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Train_D/Normal'

# 创建输出文件夹
os.makedirs(output_folder_1, exist_ok=True)
os.makedirs(output_folder_2, exist_ok=True)

# 读取标签文件
df = pd.read_excel(label_file)

# 获取标签为1的图像（D列=1）
label_1_images = df[df['D'] == 1][['Left-Fundus', 'Right-Fundus']].stack().tolist()

# 获取正常样本（N列=1且D列=0）
normal_samples = df[(df['N'] == 1) & (df['D'] == 0)]
normal_ids = normal_samples['ID'].unique().tolist()

# 构建ID到图像的映射
id_to_images = {}
for id in normal_ids:
    id_records = normal_samples[normal_samples['ID'] == id]
    id_to_images[id] = {
        'left': id_records['Left-Fundus'].iloc[0] if not id_records['Left-Fundus'].empty else None,
        'right': id_records['Right-Fundus'].iloc[0] if not id_records['Right-Fundus'].empty else None
    }

# 计算需要选择的ID数量（考虑单眼情况）
num_label_1 = len(label_1_images)
random_label_0_images = []
attempts = 0

# 动态选择直到满足数量或遍历所有ID
while len(random_label_0_images) < num_label_1 and attempts < len(normal_ids):
    random_id = random.choice(normal_ids)
    left_img = id_to_images[random_id]['left']
    right_img = id_to_images[random_id]['right']

    # 添加存在的图像
    if left_img and left_img not in random_label_0_images:
        random_label_0_images.append(left_img)
    if right_img and right_img not in random_label_0_images:
        random_label_0_images.append(right_img)

    # 去重处理
    normal_ids = list(set(normal_ids))  # 防止重复选择
    attempts += 1

# 截取至目标数量
random_label_0_images = random_label_0_images[:num_label_1]


# 复制文件函数
def safe_copy(file_list, destination):
    for file in file_list:
        src = os.path.join(train_folder, file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(destination, file))


# 执行复制操作
safe_copy(label_1_images, output_folder_1)
safe_copy(random_label_0_images, output_folder_2)

print(f"复制完成！D类图像：{len(label_1_images)}张，Normal类图像：{len(random_label_0_images)}张")