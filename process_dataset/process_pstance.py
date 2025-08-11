import pandas as pd
import glob

# 定义合并逻辑
def merge_files(pattern, output_name):
    # 匹配所有符合模式的文件（例如：所有包含 "train" 的文件）
    file_list = glob.glob(f"*{pattern}*.csv")
    # 读取并合并数据
    combined_df = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)
    # 保存结果
    combined_df.to_csv(output_name, index=False)

# 按类型合并文件
merge_files("train", "train.csv")  # 合并所有含 "train" 的文件到 train.csv
merge_files("test", "test.csv")     # 合并所有含 "test" 的文件到 test.csv
merge_files("val", "val.csv")       # 合并所有含 "val" 的文件到 val.csv