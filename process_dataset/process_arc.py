import pandas as pd

# # Load the data
# stances_bodies = pd.read_csv(r'D:\Computational Argumentation\StanceDection\dataset\ARC\raw\arc_bodies.csv')
# arc_train = pd.read_csv(r'D:\Computational Argumentation\StanceDection\dataset\ARC\raw\arc_stances_train.csv')
# arc_test = pd.read_csv(r'D:\Computational Argumentation\StanceDection\dataset\ARC\raw\arc_stances_test.csv')
#
# # Merge stances_test with arc_test and arc_train
# merged_test = pd.merge(arc_test, stances_bodies, on='Body ID', how='left')
# merged_train = pd.merge(arc_train, stances_bodies, on='Body ID', how='left')
#
# # Save the merged data to new CSV files
# merged_test.to_csv(r'D:\Computational Argumentation\StanceDection\dataset\ARC\split\test.csv', index=False)
# merged_train.to_csv(r'D:\Computational Argumentation\StanceDection\dataset\ARC\split\train.csv', index=False)
#
# print("Merged files have been saved as 'train.csv' and 'test.csv'.")

import pandas as pd

# 读取 CSV 文件
df = pd.read_csv(r'..\dataset\ARC\split\train.csv')  # 将 'your_file.csv' 替换为你的文件名

# 修改列名：将 Headline 改为 Target
df.rename(columns={'Headline': 'Target'}, inplace=True)
df.rename(columns={'articleBody': 'Sentence'}, inplace=True)
# 删除 Body ID 列
df.drop(columns=['Body ID'], inplace=True)

# 保存修改后的数据到新的 CSV 文件
df.to_csv(r'..\ARC\split\train.csv', index=False)

print("修改后的文件已保存为 'modified_file.csv'")