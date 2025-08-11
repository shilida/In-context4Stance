import pandas as pd
from sklearn.model_selection import train_test_split

# 读取TXT文件（假设文件名为input.txt）
# 注意：这里我们直接从您提供的示例数据创建DataFrame，实际使用时请替换为文件读取代码
df = pd.read_csv(r'xx\NLPCC2016\raw\evasampledata4-TaskAA.txt', sep='\t', encoding='utf-8')


# 转换为所需的3列格式
result_df = df[['TARGET', 'TEXT', 'STANCE']]
result_df.columns = ['Target', 'Sentence', 'Stance']

# 按照8:2比例分割数据集
train_df, test_df = train_test_split(result_df, test_size=0.2, random_state=42)

# 保存为CSV文件
train_df.to_csv(r'..\dataset\NLPCC2016\split\train.csv', index=False, encoding='utf-8-sig')
test_df.to_csv(r'..\dataset\NLPCC2016\split\test.csv', index=False, encoding='utf-8-sig')

print("转换完成！训练集已保存为train.csv，测试集已保存为test.csv")