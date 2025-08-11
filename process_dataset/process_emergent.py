import pandas as pd
import os
import csv
import argparse  # 导入argparse库

# 替换字典定义
emergent_label_dict = {
    "for": "for",
    "against": "against",
    "observing": "observing"
}


def write_data_files(data, output_folder):
    """
    将数据写入指定的输出文件夹。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    df = pd.DataFrame(data)
    print(f'=== Writing to: {output_folder} ====')
    print(f'=== Before filtering empty texts: {len(df)}')
    df = df[df["text"] != ""].dropna(subset=['text'])  # 增加dropna以确保安全
    print(f'=== After filtering empty texts: {len(df)}')

    output_path = os.path.join(output_folder, 'data.jsonl')
    df.to_json(output_path, lines=True, orient='records')
    print(f'=== Data successfully written to {output_path} ===')


def preprocess_emergent(input_folder, output_folder, label_dict):
    """
    预处理 'emergent' 数据集。

    Args:
        input_folder (str): 包含原始数据文件（.csv）的文件夹路径。
        output_folder (str): 用于保存处理后文件的文件夹路径。
        label_dict (dict): 标签映射字典。
    """
    print(f"\n===================== Start preprocessing files in: {input_folder} =====================")

    # 使用os.path.join构建跨平台兼容的路径
    split_file_path = os.path.join(input_folder, "emergent_splits.csv")
    data_file_path = os.path.join(input_folder, "url-versions-2015-06-14-clean.csv")

    try:
        split_file = pd.read_csv(split_file_path)
        data_file = pd.read_csv(data_file_path)
    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}")
        print(
            "Please ensure your input directory contains 'emergent_splits.csv' and 'url-versions-2015-06-14-clean.csv'")
        return

    split_file.set_index("claimId", inplace=True)
    data = []
    for item in data_file.itertuples():
        # 添加健壮性检查，以防claimId不存在
        if item.claimId in split_file.index:
            data.append({
                'text': str(item.articleHeadline),  # 确保是字符串
                'label': label_dict.get(item.articleHeadlineStance, "unknown"),  # 使用.get避免KeyError
                'target': str(item.claimHeadline),
                'split': split_file.loc[item.claimId].split
            })
        else:
            print(f"Warning: claimId {item.claimId} not found in split file. Skipping.")

    print(f"Total processed records: {len(data)}")
    write_data_files(data, output_folder)


if __name__ == '__main__':
    # 1. 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(description="Preprocess the 'emergent' dataset.")

    # 2. 添加输入和输出路径参数
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Path to the raw 'emergent' dataset directory (containing the csv files).")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path to the directory where the processed files will be saved.")

    # 3. 解析命令行传入的参数
    args = parser.parse_args()

    # 4. 使用解析后的参数调用主函数
    preprocess_emergent(args.input_dir, args.output_dir, emergent_label_dict)

    # 之前被注释掉的个人路径代码已被完全移除，以保持整洁。