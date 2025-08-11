import os
import pandas as pd

# 定义主题列表
topics = ["abortion", "gun_control"]

# 定义合并后的文件路径
output_files = {
    "train": "train.csv",
    "dev": "dev.csv",
    "test": "test.csv"
}

# 初始化空的DataFrame用于存储合并后的数据
merged_data = {
    "train": pd.DataFrame(),
    "dev": pd.DataFrame(),
    "test": pd.DataFrame()
}

# 遍历每个主题
for topic in topics:
    # 遍历每个文件类型（train, dev, test）
    for file_type in output_files.keys():
        # 构建文件路径
        file_path = f"{topic}_{file_type}.csv"

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，跳过。")
            continue

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 添加主题列
        df["target"] = topic

        # 选择需要的列并重命名
        df = df.rename(columns={
            "tweet_text": "tweet",  # 将tweet_text重命名为tweet
            "stance": "stance"  # stance列保持不变
        })
        df = df[["tweet", "target", "stance"]]  # 只保留需要的列

        # 将数据添加到合并后的DataFrame中
        merged_data[file_type] = pd.concat([merged_data[file_type], df], ignore_index=True)

# 保存合并后的文件
for file_type, output_path in output_files.items():
    if not merged_data[file_type].empty:
        merged_data[file_type].to_csv(output_path, index=False)
        print(f"已保存合并后的文件：{output_path}")
    else:
        print(f"没有数据可保存到 {output_path}")

print("合并完成！")