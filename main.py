import asyncio
import pandas as pd
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sklearn.metrics import classification_report
import argparse
from utils import *
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms.base import LLM
import preprocessor
import os
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION,
                         preprocessor.OPT.HASHTAG)
from langsmith import wrappers, traceable
# 设置环境变量 (建议在运行环境外部设置，而不是硬编码在代码中)
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_API_KEY'] = 'YOUR_LANGCHAIN_API_KEY'  # <-- 已替换为占位符
# os.environ['LANGCHAIN_PROJECT'] = 'StanceClassification'
import random
import warnings
from tqdm.asyncio import tqdm_asyncio  # 使用 tqdm 的异步支持
# from langchain_openai import AzureOpenAI
import requests
import csv
from tqdm import tqdm
from icl import select_train_samples


class ResultSchema(BaseModel):
    Stance: str = Field(
        description="")
    # Think_proecess: str = Field(description="Your thought process")


async def invoke_llm_with_retry(llm, system_message, messages, max_retries=5, delay=10):
    retries = 0
    # print([SystemMessage(content=system_message), HumanMessage(content=messages)])
    while retries < max_retries:
        try:
            output = await llm.ainvoke([SystemMessage(content=system_message), HumanMessage(content=messages)])
            return output
        except Exception as e:
            retries += 1
            print(f"调用失败，重试 {retries}/{max_retries}... 错误信息: {e}")
            if retries < max_retries:
                await asyncio.sleep(delay)
            else:
                raise


def get_label_index(label, label_to_index):
    """
    根据输入的标签返回对应的索引。如果标签不存在，返回随机索引并发出警告。

    参数:
    label (str): 要查找索引的标签。
    label_to_index (dict): 标签到索引的映射字典。

    返回:
    int: 标签对应的索引；如果标签不存在，返回随机索引。
    """
    if label in label_to_index:
        return label_to_index[label]
    else:
        # 发出警告
        warnings.warn(f"标签 '{label}' 不存在，返回随机索引。", UserWarning)
        # 返回随机索引
        return random.choice(list(label_to_index.values()))


async def process_row_async(llm, row, unique_elements_string, shot, train_df, method):
    sentence = row['Sentence']
    stance_label = row['Stance']
    topic = row['Target']
    parser = PydanticOutputParser(pydantic_object=ResultSchema)
    ResultSchema.model_fields["Stance"].description = f"Candidate stance: {unique_elements_string}."
    format_instructions = parser.get_format_instructions()
    if method == 'random' and shot != 0:
        examples = await select_train_samples(train_df, shot, row['random_indices'])
    # elif method == 'same_target':
    #     examples = await select_train_samples(train_df, shot, row['top10_same_target_indices'])
    elif method == 'global' and shot != 0:
        examples = await select_train_samples(train_df, shot, row['global_indices'])
    else:
        # print("no method")
        examples = ""
    system_text = f"""
    You are a professional stance identification expert, skilled in multi-dimensional semantic analysis and dialectical reasoning to determine the stance of a text towards a specified target.
    Note: You can only choose one of the {unique_elements_string}.
    Output format instructions: {format_instructions}
    {examples}
    """
    messages_text = f"""
    Text:{sentence}
    Target:{topic}
    Stance:?
    """
    retries = 0
    pred_stance = ""
    think_proecess = ""
    while retries < 5:
        try:
            output = await invoke_llm_with_retry(llm, system_text, messages_text)
            parsed_output = parser.parse(output.content)
            pred_stance = parsed_output.Stance
            # think_proecess = parsed_output.Think_proecess
            return sentence, topic, pred_stance, stance_label, think_proecess
        except Exception as e:
            # print(f"解析失败: {e}, 重试次数: {retries + 1}")
            print(f"解析失败:, 重试次数: {retries + 1}")
            retries += 1
            await asyncio.sleep(3)  # 避免频繁重试
    # 达到最大重试次数后返回默认值
    print(f"达到最大重试次数 {5}, 返回默认值")
    return sentence, topic, pred_stance, stance_label, think_proecess


async def limited_task(task, semaphore):
    async with semaphore:
        return await task


async def batch_process(df, process_func, batch_size=100, concurrency_limit=50):
    results = []
    total_tasks = len(df)
    semaphore = asyncio.Semaphore(concurrency_limit)  # 全局信号量
    progress_bar = tqdm(total=total_tasks, desc="整体进度", unit="任务")

    for batch_idx in range(0, total_tasks, batch_size):
        # 动态生成批次任务
        batch_df = df.iloc[batch_idx:batch_idx + batch_size]

        # 创建当前批次任务
        tasks = [process_func(row) for _, row in batch_df.iterrows()]

        # 限制并发执行
        processed_tasks = [limited_task(task, semaphore) for task in tasks]
        batch_results = await asyncio.gather(*processed_tasks)

        results.extend(batch_results)
        progress_bar.update(len(batch_df))
        print(f"已完成批次 {batch_idx // batch_size + 1}/{(total_tasks + batch_size - 1) // batch_size}")

    progress_bar.close()
    return results


async def process_data_async(df, train_df, dataset, llm_name, method, shot, head, temp):
    llm = ChatOpenAI(
        openai_api_base="YOUR_API_BASE_URL",
        model_name=llm_name,
        temperature=temp,
        openai_api_key="YOUR_API_KEY"
    )

    # 预处理数据
    label_to_index = {label: idx for idx, label in enumerate(df['Stance'].unique())}
    unique_elements_string = ';'.join(df['Stance'].unique())

    # 定义行处理函数（闭包保持llm状态）
    async def process_row_wrapper(row):
        return await process_row_async(llm, row, unique_elements_string, shot, train_df, method)

    # 分批处理任务
    arg_stances = await batch_process(
        df=df,
        process_func=process_row_wrapper,
        batch_size=100,
        concurrency_limit=100
    )
    pred_stances_list = []
    label_stances_list = []
    for case in arg_stances:
        pred_stance = get_label_index(case[2], label_to_index)
        label_stance = get_label_index(case[3], label_to_index)
        pred_stances_list.append(pred_stance)
        label_stances_list.append(label_stance)
    stance_metrics = classification_report(label_stances_list, pred_stances_list, output_dict=True, digits=4)
    weighted_stance_f1 = round(stance_metrics["weighted avg"]['f1-score'], 4)
    macro_stance_f1 = round(stance_metrics["macro avg"]['f1-score'], 4)
    stance_acc = round(stance_metrics["accuracy"], 4)

    # 文件路径是相对路径，不包含个人信息，因此予以保留
    csv_file = "./experiments/{}/{}_{}shot_head{}_f1{}.csv".format(llm_name, dataset, shot, head, weighted_stance_f1)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence', 'Target', 'Pred_stance', 'Label_stance', "Think_process"])
        for row in arg_stances:
            writer.writerow(row)
    print('weighted_f1_{}'.format(dataset), weighted_stance_f1)
    print('macro_f1_{}'.format(dataset), macro_stance_f1)
    print('acc_{}'.format(dataset), stance_acc)
    return weighted_stance_f1, macro_stance_f1, stance_acc


if __name__ == '__main__':
    for shot in [0, 1, 2, 3, 5, 10]:
        if shot == 0:
            methods = ["random"]
        else:
            methods = ["random", "global"]
        for mehtod in methods:
            parser = argparse.ArgumentParser()
            parser.add_argument("--llm_name", type=str, default='qwen2.5-72b',
                                choices=['deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'gpt-3.5-turbo-0125',
                                         'gemini-1.5-pro', 'Llama-3.1-70B-Instruct', 'qwen2.5-72b'])
            parser.add_argument("--shot", type=int, default=shot, help="shot")
            parser.add_argument('--method', type=str, default=mehtod)
            parser.add_argument('--head', type=int, default=0, help="head number", )
            parser.add_argument('--temp', type=float, default=0, help="temp")
            args = parser.parse_args()
            n = 3
            llm_name = args.llm_name
            for dataset in ['ArgMin', 'SemEval2016', 'NLPCC2016', 'imgArg', 'VAST', 'covid-19-tweets', 'IBM-CSD',
                            'PStance-main', 'perspectrum', 'emergent', 'poldeb', 'ARC', 'FNC-1']:
                macro_f1_list = []
                acc_list = []
                for i in range(n):
                    print("-----------------------{}_{}_{}_{}-----------------------".format(dataset, mehtod, shot, i))
                    test_df = pd.read_csv('./dataset/{}/split/test_with_in_domain.csv'.format(dataset), index_col=None)
                    train_df = pd.read_csv('./dataset/{}/split/train.csv'.format(dataset), index_col=None)
                    test_df = test_df.dropna(axis=0, how="any")
                    if args.head != 0:
                        weighted_stance_f1, macro_stance_f1, stance_acc = asyncio.run(
                            process_data_async(test_df.head(args.head), train_df, dataset, llm_name, args.method,
                                               shot=args.shot,
                                               head=args.head,
                                               temp=args.temp))
                    else:
                        weighted_stance_f1, macro_stance_f1, stance_acc = asyncio.run(
                            process_data_async(test_df, train_df, dataset, llm_name, args.method, shot=args.shot,
                                               head=args.head,
                                               temp=args.temp
                                               ))
                    macro_f1_list.append(macro_stance_f1)
                    acc_list.append(stance_acc)
                    print('macro_stance_f1_{}'.format(dataset), macro_stance_f1)
                    print('stance_acc_{}'.format(dataset), stance_acc)
                print("macro_f1_list", macro_f1_list)
                print("acc_list", acc_list)
                mean_f1, variance_f1 = list_stats(macro_f1_list)
                mean_acc, variance_acc = list_stats(acc_list)

                print('mean_macro_stance_f1_{}_shot_{}_method_{}_temp_{}'.format(dataset, shot, args.method, args.temp),
                      mean_f1)
                print('mean_stance_acc_{}_shot_{}_method_{}_temp_{}'.format(dataset, shot, args.method, args.temp),
                      mean_acc)
                print("variance_acc_{}_shot_{}_method_{}_temp_{}".format(dataset, shot, args.method, args.temp),
                      variance_acc)
                print("variance_f1_{}_shot_{}_method_{}_temp_{}".format(dataset, shot, args.method, args.temp),
                      variance_f1)
