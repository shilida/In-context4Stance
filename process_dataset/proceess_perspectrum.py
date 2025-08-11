import json
import os
perspectrum_label_dict = {
    "SUPPORT": "SUPPORT",
    "UNDERMINE": "UNDERMINE",
    # "SUPPORT": 0,
    # "UNDERMINE": 1
}
import pandas as pd
def preprocess_perspectrum(file, output_folder, label_dict):
    # split based on Schiller SD benchmark
    def read_and_tokenize_csv(file, label_dict):

        with open(file+"dataset_split_v1.0.json", "r") as split_in, \
                open(file+"perspectrum_with_answers_v1.0.json","r") as claims_in, \
                open(file+"perspective_pool_v1.0.json", "r") as perspectives_in:

            # load files
            data_split = json.load(split_in)
            claims = json.load(claims_in)
            perspectives = json.load(perspectives_in)

            # lookup for perspective ids
            perspectives_dict = {}
            for p in perspectives:
                perspectives_dict[p['pId']] = p['text']

            # init
            X_train, X_dev, X_test = [], [], []
            y_train, y_dev, y_test = [], [], []

            count_sup_cluster = 0
            count_und_cluster = 0

            # fill train/dev/test
            for claim in claims:
                cId = str(claim['cId'])
                for p_cluster in claim['perspectives']:
                    cluster_label = label_dict[p_cluster['stance_label_3']]
                    for pid in p_cluster['pids']:
                        if data_split[cId] == 'train':
                            X_train.append((claim['text'], perspectives_dict[pid]))
                            y_train.append(cluster_label)
                        elif data_split[cId] == 'dev':
                            X_dev.append((claim['text'], perspectives_dict[pid]))
                            y_dev.append(cluster_label)
                        elif data_split[cId] == 'test':
                            X_test.append((claim['text'], perspectives_dict[pid]))
                            y_test.append(cluster_label)
                        else:
                            print("Incorrect set type: "+data_split[claim['cId']])
                    if cluster_label == 1:
                        count_sup_cluster += 1
                    if cluster_label == 0:
                        count_und_cluster += 1

        return X_train, y_train, X_dev, y_dev, X_test, y_test

    print("\n===================== Start preprocessing file: "+ file + " =====================")

    # read and tokenize data
    X_train, y_train, X_dev, y_dev, X_test, y_test = read_and_tokenize_csv(file, label_dict)

    assert len(X_train) == 6978
    assert len(X_dev) == 2071
    assert len(X_test) == 2773

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)
def write_data_files(data, output_folder):
    df = pd.DataFrame(data)
    print('=== Writing to: ' + output_folder + ' ====')
    print('=== Before filtering empty texts: ' + str(len(df)))
    df = df[df["text"] != ""]
    print('=== After filtering empty texts: ' + str(len(df)))
    df.to_json(os.path.join(output_folder, 'data.jsonl'), lines=True, orient='records')
def get_raw():
# perspectrum
    perspectrum_path = 'source_path'
    perspectrum_output_path = r"dataset\perspectrum\raw"
    if not os.path.exists(perspectrum_output_path):
        os.makedirs(perspectrum_output_path)
    preprocess_perspectrum(perspectrum_path, perspectrum_output_path, perspectrum_label_dict)


import json
import csv


def jsonl_to_csv(jsonl_file, train_file, test_file, dev_file):
    # 打开CSV文件并写入列标题
    with open(train_file, 'w', newline='', encoding='utf-8') as train_csv, \
            open(test_file, 'w', newline='', encoding='utf-8') as test_csv, \
            open(dev_file, 'w', newline='', encoding='utf-8') as dev_csv:

        # 创建CSV写入器
        train_writer = csv.writer(train_csv)
        test_writer = csv.writer(test_csv)
        dev_writer = csv.writer(dev_csv)

        # 写入列标题
        train_writer.writerow(['Sentence', 'Stance', 'Target'])
        test_writer.writerow(['Sentence', 'Stance', 'Target'])
        dev_writer.writerow(['Sentence', 'Stance', 'Target'])

        # 读取JSONL文件
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                # 根据split字段写入不同的CSV文件
                if data['split'] == 'train':
                    train_writer.writerow([data['text'], data['label'], data['target']])
                elif data['split'] == 'test':
                    test_writer.writerow([data['text'], data['label'], data['target']])
                elif data['split'] == 'dev':
                    dev_writer.writerow([data['text'], data['label'], data['target']])

if __name__ == '__main__':
    get_raw()
    # 示例用法
    jsonl_file = r'../dataset\perspectrum\data.jsonl'
    train_file = r'../dataset\perspectrum\split\train.csv'
    test_file = r'../dataset\perspectrum\split\test.csv'
    dev_file = r'../dataset\perspectrum\split\dev.csv'
    # 调用函数进行转换
    jsonl_to_csv(jsonl_file, train_file, test_file,dev_file)

    print("JSONL文件已成功转换为CSV文件！")
