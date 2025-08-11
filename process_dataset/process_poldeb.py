import pandas as pd
import os
import csv
poldeb_label_dict = {
    "stance1": 'Pro', # pro
    "stance2": 'Con' # con
}
def write_data_files(data, output_folder):
    df = pd.DataFrame(data)
    print('=== Writing to: ' + output_folder + ' ====')
    print('=== Before filtering empty texts: ' + str(len(df)))
    df = df[df["text"] != ""]
    print('=== After filtering empty texts: ' + str(len(df)))
    df.to_json(os.path.join(output_folder, 'data.jsonl'), lines=True, orient='records')

def preprocess_poldeb(file, output_folder, label_dict):
    print("\n===================== Start preprocessing file: "+ file + " =====================")
    # the split is based on the paper "Cross-Domain Label-Adaptive Stance Detection" by Hardalov et al. 2021
    target_folder_mapping = {
        'abortion': 'abortion',
        'creation': 'creationism vs evolution in schools',
        'gayRights': 'adoption of children by same-sex couples',
        'god': 'is there a god',
        'guns': 'armed pilots in airplanes',
        'healthcare': 'public insurance option in US health care'
    }
    dir_listing = os.listdir(file)
    target_folders = [i for i in dir_listing if i in target_folder_mapping]

    def read_post(filepath):
        with open(filepath, 'r', errors='ignore') as fp:
            lines = fp.readlines()
            stance = lines[0].strip().split('=')[1]
            text = lines[3].strip()
            return stance, text

    X_train, y_train, X_dev, y_dev, X_test, y_test = [], [], [], [], [], []
    for folder in target_folders:
        for post in os.listdir(os.path.join(file,folder)):
            stance, text = read_post(os.path.join(file,folder,post))
            if folder in ['healthcare', 'guns', 'gayRights', 'god']:
                X_train.append((target_folder_mapping[folder], text))
                y_train.append(label_dict[stance])
            if folder in ['abortion']:
                X_dev.append((target_folder_mapping[folder], text))
                y_dev.append(label_dict[stance])
            if folder in ['creation']:
                X_test.append((target_folder_mapping[folder], text))
                y_test.append(label_dict[stance])

    assert len(X_train) == 4753
    assert len(X_dev) == 1151
    assert len(X_test) == 1230

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x, y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)
from process_iac import jsonl_to_csv
if __name__ == '__main__':
    # poldeb
    poldeb_path = r'xxxx'
    poldeb_output_path = r'xxxx'
    if not os.path.exists(poldeb_output_path):
        os.makedirs(poldeb_output_path)
    preprocess_poldeb(poldeb_path, poldeb_output_path, poldeb_label_dict)
    jsonl_file = r'..\dataset\poldeb\raw\data.jsonl'  # JSONL文件路径
    train_file = r'..\dataset\poldeb\split\train.csv'  # 训练集CSV文件路径
    test_file = r'..\dataset\poldeb\split\test.csv'  # 测试集CSV文件路径
    dev_file = r'..\dataset\poldeb\split\dev.csv'  # 开发集CSV文件路径
    # 调用函数进行转换
    jsonl_to_csv(jsonl_file, train_file, test_file, dev_file)