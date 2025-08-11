import pandas as pd
import os
import csv
def get_csv():
    # 定义tsv文件所在的目录
    directory = 'dataset/ArgMin/raw_data'  # 替换为你的tsv文件所在的目录路径

    # 指定需要提取的列
    columns_to_extract = ['topic', 'sentence', 'annotation', 'set']

    # 创建一个空的DataFrame用于存储所有数据
    all_data = pd.DataFrame(columns=columns_to_extract)

    # 遍历目录中的所有tsv文件
    for filename in os.listdir(directory):
        if filename.endswith('.tsv'):  # 确保只处理tsv文件
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    rows = []
                    for row in reader:
                        try:
                            # 只提取指定的列
                            extracted_row = {col: row[col] for col in columns_to_extract}
                            rows.append(extracted_row)
                        except KeyError:
                            print(f"警告：文件 {filename} 中缺少某些列，跳过该行。")
                    # 将有效行转换为DataFrame并追加到all_data
                    temp_df = pd.DataFrame(rows)
                    all_data = pd.concat([all_data, temp_df], ignore_index=True)
            except Exception as e:
                print(f"警告：文件 {filename} 出现错误 {e}，跳过该文件。")

    # 根据'set'列的值进行分组
    train_data = all_data[all_data['set'] == 'train']
    val_data = all_data[all_data['set'] == 'val']
    test_data = all_data[all_data['set'] == 'test']

    # 将每个部分保存为csv文件，只包含指定的列
    train_data.to_csv('train.csv', index=False, columns=columns_to_extract)
    val_data.to_csv('val.csv', index=False, columns=columns_to_extract)
    test_data.to_csv('test.csv', index=False, columns=columns_to_extract)

    print("文件已成功拆分为 train.csv, val.csv 和 test.csv，且只包含指定的列")
def change_label():
    # 读取CSV文件
    file_path = '../dataset/ArgMin/split/test.csv'  # 输入文件路径
    data = pd.read_csv(file_path)  # 假设文件是TSV格式，分隔符为制表符

    # 定义映射关系
    stance_mapping = {
        'NoArgument': 'NONE',
        'Argument_against': 'AGAINST',
        'Argument_for': 'FAVOR'
    }

    # 使用map函数替换Stance列的值
    data['Stance'] = data['Stance'].map(stance_mapping)

    # 将修改后的数据保存为新的CSV文件
    output_file = '../dataset/ArgMin/split/test.csv'  # 输出文件路径
    data.to_csv(output_file, index=False)  # 保存为TSV文件

    print(f"数据已成功修改并保存为 {output_file}")
if __name__ == '__main__':
    change_label()