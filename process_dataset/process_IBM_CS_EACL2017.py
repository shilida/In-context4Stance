import csv

# 输入文件路径
input_file = "../dataset/IBM-CSD/raw/claim_stance_dataset_v1.csv"

# 输出文件路径
train_file = "../dataset/IBM-CSD/split/train.csv"
test_file = "../dataset/IBM-CSD/split/test.csv"

# 打开输入文件并读取数据
with open(input_file, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)

    # 初始化训练集和测试集
    train_data = []
    test_data = []

    # 遍历每一行
    for row in reader:
        # 提取所需字段
        topic_text = row['topicText']
        claim_corrected_text = row['claims.claimCorrectedText']
        stance = row['claims.stance']

        # 根据 split 字段将数据分为训练集和测试集
        if row['split'] == 'train':
            train_data.append([topic_text, claim_corrected_text, stance])
        elif row['split'] == 'test':
            test_data.append([topic_text, claim_corrected_text, stance])

# 写入训练集文件
with open(train_file, mode='w', newline='', encoding='utf-8') as train_outfile:
    writer = csv.writer(train_outfile)
    writer.writerow(['topicText', 'claimCorrectedText', 'stance'])  # 写入表头
    writer.writerows(train_data)  # 写入数据

# 写入测试集文件
with open(test_file, mode='w', newline='', encoding='utf-8') as test_outfile:
    writer = csv.writer(test_outfile)
    writer.writerow(['topicText', 'claimCorrectedText', 'stance'])  # 写入表头
    writer.writerows(test_data)  # 写入数据

print(f"训练集已保存到 {train_file}")
print(f"测试集已保存到 {test_file}")