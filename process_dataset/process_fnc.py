import json
import csv
import os
import argparse

# 严格定义允许的标签（小写形式）
VALID_LABELS = {"agree", "disagree", "unrelated", "discuss"}


def clean_label(raw_label: str) -> str:
    """
    标准化并验证标签。
    将标签去除首尾空格、转为小写，并检查是否在有效标签集合中。

    Args:
        raw_label (str): 原始标签字符串。

    Returns:
        str: 清理后的有效标签，如果无效则返回 None。
    """
    if not isinstance(raw_label, str):
        return None
    processed = raw_label.strip().lower()
    return processed if processed in VALID_LABELS else None


def jsonl_to_csv(jsonl_file: str, train_file: str, test_file: str, dev_file: str):
    """
    读取一个 JSONL 文件，并根据其中的 'split' 字段将其内容分发到
    train, test, 和 dev 三个 CSV 文件中。

    Args:
        jsonl_file (str): 输入的 JSONL 文件路径。
        train_file (str): 输出的训练集 CSV 文件路径。
        test_file (str): 输出的测试集 CSV 文件路径。
        dev_file (str): 输出的开发集 CSV 文件路径。
    """
    # 确保输出文件的目录存在
    for f in [train_file, test_file, dev_file]:
        os.makedirs(os.path.dirname(f), exist_ok=True)

    # 初始化CSV文件
    try:
        with open(train_file, 'w', newline='', encoding='utf-8') as train_csv, \
                open(test_file, 'w', newline='', encoding='utf-8') as test_csv, \
                open(dev_file, 'w', newline='', encoding='utf-8') as dev_csv, \
                open(jsonl_file, 'r', encoding='utf-8') as file:

            # 创建CSV写入器
            writers = {
                "train": csv.writer(train_csv),
                "test": csv.writer(test_csv),
                "dev": csv.writer(dev_csv)
            }

            # 写入列标题
            header = ['Sentence', 'Stance', 'Target']
            for writer in writers.values():
                writer.writerow(header)

            print(f"开始处理文件: {jsonl_file}")

            # 处理JSONL文件
            for line_num, line in enumerate(file, 1):
                try:
                    # 跳过空行
                    if not line.strip():
                        continue

                    data = json.loads(line)

                    # 1. 强制字段检查
                    required_fields = ['text', 'label', 'target', 'split']
                    for field in required_fields:
                        if field not in data:
                            raise ValueError(f"缺失必要字段 '{field}'")

                    # 2. 严格标签过滤
                    raw_label = data['label']
                    clean_label_value = clean_label(raw_label)
                    if not clean_label_value:
                        print(f"警告: 第 {line_num} 行: 无效标签 '{raw_label}' (已跳过)")
                        continue

                    # 3. 检查split值有效性
                    split = data.get('split', '').lower()
                    if split not in writers:
                        print(f"警告: 第 {line_num} 行: 未知split值 '{split}' (已跳过)")
                        continue

                    # 4. 写入CSV
                    writers[split].writerow([
                        data['text'],
                        clean_label_value,
                        data['target']
                    ])

                except json.JSONDecodeError:
                    print(f"错误: 第 {line_num} 行: JSON解析失败 (已跳过)")
                except Exception as e:
                    print(f"错误: 第 {line_num} 行: 处理失败 - {e}")

            print("文件处理完成。")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{jsonl_file}'")
    except Exception as e:
        print(f"发生未知错误: {e}")


def main():
    """主函数，用于解析命令行参数并调用处理函数。"""
    parser = argparse.ArgumentParser(
        description="将包含 'split' 字段的 JSONL 文件转换为 train/test/dev 三个 CSV 文件。"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入的 JSONL 文件路径。"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="保存输出的 train.csv, test.csv, dev.csv 文件的目录路径。"
    )

    args = parser.parse_args()

    # 构建输出文件的完整路径
    train_path = os.path.join(args.output_dir, 'train.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')
    dev_path = os.path.join(args.output_dir, 'dev.csv')

    # 调用转换函数
    jsonl_to_csv(args.input_file, train_path, test_path, dev_path)


if __name__ == '__main__':
    main()