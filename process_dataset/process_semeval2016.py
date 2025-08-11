import pandas as pd

# 读取上传的txt文件
file_path = 'dataset/SemEval2016/stance-data-all-annotations/data-all-annotations/trainingdata-all-annotations.txt'  # 文件路径
# 使用pandas读取TSV文件，并指定编码格式为latin1或ISO-8859-1
data = pd.read_csv(file_path, sep='\t', usecols=['Target', 'Tweet', 'Stance'], encoding='latin1')

# 将提取的数据保存为CSV文件
output_file = '../dataset/SemEval2016/split/train.csv'  # 输出文件名
data.to_csv(output_file, index=False, encoding='utf-8')  # 保存时仍使用UTF-8编码

print(f"数据已成功提取并保存为 {output_file}")