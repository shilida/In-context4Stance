import asyncio
import base64
from PIL import Image
import io
import pandas as pd
import preprocessor
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import random
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sklearn.metrics import classification_report
import argparse
from utils import *
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms.base import LLM
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION,
                         preprocessor.OPT.HASHTAG)
import random
import warnings
from tqdm.asyncio import tqdm_asyncio  # 使用 tqdm 的异步支持
# from langchain_openai import AzureOpenAI
import requests
import csv
import os
from tqdm import tqdm

from icl import select_train_samples


class ResultSchema(BaseModel):
    Stance: str = Field(
        description="")
async def invoke_llm_with_retry(llm, system_message, messages, max_retries=5, delay=10):
    retries = 0
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
    if method == 'random':
        examples = await select_train_samples(train_df, shot, row['random_indices'])
    elif method == 'global':
        examples = await select_train_samples(train_df, shot, row['top30_global_indices'])
    else:
        # print("no method")
        examples = ""
    # system_text = f"""
    # You are a professional cross-domain stance identification expert, skilled in multi-dimensional semantic analysis and dialectical reasoning to determine the stance of a text towards a specified target.
    # Note: You can only choose one of the {unique_elements_string}.
    # Output format instructions: {format_instructions}
    # """
    system_text = f"""
    You are a professional cross-domain stance identification expert, skilled in multi-dimensional semantic analysis and dialectical reasoning to determine the stance of a text towards a specified target.
    Note: You can only choose one of the {unique_elements_string}.
    Output format instructions: {format_instructions}
    {examples}
    The current examples are sourced from cross-domain stance classification datasets, where the original labeling schema may not align with the annotation guidelines of the current target task.
    """
    messages_text = f"""
    Text:{sentence}
    Target:{topic}
    Stance:?
    Note: You can only choose one of the {unique_elements_string}.
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
            await asyncio.sleep(1)  # 避免频繁重试
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


async def process_data_async(df, train_df, dataset1, dataset2_name, llm_name, method, shot, head):
    # 初始化自定义LLM实例
    llm = ChatOpenAI(
        openai_api_base="YOUR_API_BASE_URL",
        model_name=llm_name,
        openai_api_key="YOUR_API_KEY"
    )
    label_to_index = {label: idx for idx, label in enumerate(df['Stance'].unique())}
    unique_elements_string = ';'.join(df['Stance'].unique())
    async def process_row_wrapper(row):
        return await process_row_async(llm, row, unique_elements_string, shot, train_df, method)
    # 分批处理任务
    arg_stances = await batch_process(
        df=df,
        process_func=process_row_wrapper,
        batch_size=1000,  # 适当增大批次
        concurrency_limit=1000  # 降低并发数
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
    csv_file = "./experiments/{}/{}_cross_{}_{}shot_head{}_f1{}.csv".format(llm_name, dataset1, dataset2_name, shot,
                                                                            head, weighted_stance_f1)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence', 'Target', 'Pred_stance', 'Label_stance', "Think_process"])
        for row in arg_stances:
            writer.writerow(row)
    print('weighted_f1_{}_cross_{}'.format(dataset1, dataset2_name), weighted_stance_f1)
    print('macro_f1_{}_cross_{}'.format(dataset1, dataset2_name), macro_stance_f1)
    print('acc_{}_cross_{}'.format(dataset1, dataset2_name), stance_acc)
    return weighted_stance_f1, macro_stance_f1, stance_acc
def exclude_element(lst, element):
    """排除指定元素的列表"""
    return [x for x in lst if x != element]
if __name__ == '__main__':
    for shot in [1, 3, 5]:
        methods = ["random", "global"]
        for mehtod in methods:  # "random","global"
            parser = argparse.ArgumentParser()
            parser.add_argument("--llm_name", type=str, default='qwen2.5-72b',
                                choices=['deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
                                         'Llama-3.1-70B-Instruct', 'qwen2.5-72b'])
            parser.add_argument("--shot", type=int, default=shot, help="shot")  # 接下来要跑shot 1 global
            parser.add_argument('--method', type=str, default=mehtod, help="random",
                                choices=['random', 'global', 'same_target'])
            parser.add_argument('--temp', type=float, default=0, help="temp")
            parser.add_argument('--head', type=int, default=0, help="head number", )
            args = parser.parse_args()
            llm_name = args.llm_name
            dataset_list = ['ArgMin', 'SemEval2016', 'VAST', 'imgArg']
            n = 3
            for dataset1 in dataset_list:  # 'FNC-1','IAC'
                exclude_list = exclude_element(dataset_list, dataset1)
                for dataset2_name in tqdm(exclude_list):
                    macro_f1_list = []
                    acc_list = []
                    for i in range(n):
                        print("-----------------------{}_{}_{}_{}_{}-----------------------".format(dataset1,
                                                                                                    dataset2_name,
                                                                                                    mehtod, shot, i))
                        test_df = pd.read_csv(
                            './dataset/{}/split/test_with_cross_{}.csv'.format(dataset1, dataset2_name), index_col=None)
                        train_df = pd.read_csv('./dataset/{}/split/train.csv'.format(dataset2_name), index_col=None)
                        test_df = test_df.dropna(axis=0, how="any")
                        test_df["Sentence"] = test_df["Sentence"].apply(lambda x: preprocessor.clean(x))
                        train_df["Sentence"] = train_df["Sentence"].apply(lambda x: preprocessor.clean(x))
                        if args.head != 0:
                            weighted_stance_f1, macro_stance_f1, stance_acc = asyncio.run(
                                process_data_async(test_df.head(args.head), train_df, dataset1, dataset2_name, llm_name,
                                                   args.method,
                                                   shot=args.shot,
                                                   head=args.head))
                        else:
                            weighted_stance_f1, macro_stance_f1, stance_acc = asyncio.run(
                                process_data_async(test_df, train_df, dataset1, dataset2_name, llm_name, args.method,
                                                   shot=args.shot,
                                                   head=args.head))
                        macro_f1_list.append(macro_stance_f1)
                        acc_list.append(stance_acc)
                        print('macro_stance_f1_{}_cross_{}'.format(dataset1, dataset2_name), macro_stance_f1)
                        print('stance_acc_{}_cross_{}'.format(dataset1, dataset2_name), stance_acc)
                    print("macro_f1_list", macro_f1_list)
                    print("acc_list", acc_list)
                    mean_f1, variance_f1 = list_stats(macro_f1_list)
                    mean_acc, variance_acc = list_stats(acc_list)
                    print('mean_macro_stance_f1_{}_cross_{}_shot_{}_method_{}_temp_{}'.format(dataset1, dataset2_name,
                                                                                              shot, args.method,
                                                                                              args.temp), mean_f1)
                    print('mean_stance_acc_{}_cross_{}_shot_{}_method_{}_temp_{}'.format(dataset1, dataset2_name, shot,
                                                                                         args.method, args.temp),
                          mean_acc)
                    print("variance_acc_{}_cross_{}_shot_{}_method_{}_temp_{}".format(dataset1, dataset2_name, shot,
                                                                                      args.method, args.temp),
                          variance_acc)
                    print("variance_f1_{}_cross_{}_shot_{}_method_{}_temp_{}".format(dataset1, dataset2_name, shot,
                                                                                     args.method, args.temp),
                          variance_f1)