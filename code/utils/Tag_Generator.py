import pandas as pd
# 读取数据
df = pd.read_csv("C:/Users/SaintCHEN/Desktop/FutureEyes/outputs/SaintCHEN_ODIR.csv")
df2 = pd.read_csv("C:/Users/SaintCHEN/Desktop/FutureEyes/outputs/Saint_ODIR_Labels.csv")
# 定义类别列
categories = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
# 遍历每一行并更新每个类别的值
for idx, row in df.iterrows():
    for col in categories:
        df2.at[idx, col] = 1 if row[col] > 0.5 else 0
# 保存到新的 CSV 文件
df2.to_csv("Saint_ODIR_Labels.csv", index=False)
