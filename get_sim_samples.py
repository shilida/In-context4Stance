import preprocessor
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION,
                         preprocessor.OPT.HASHTAG)
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from tqdm import tqdm
# 初始化 OpenAI 客户端
client = OpenAI(
    api_key='xxxxxxx',
    base_url="xxxxxxx"
)
def get_embeddings(texts, batch_size=10):
    """获取一组文本的嵌入向量，分批调用 API"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]  # 每批最多 batch_size 条文本
        response = client.embeddings.create(
            model="text-embedding-v3",
            input=batch,
            dimensions=1024,  # 根据需要调整维度
            encoding_format="float"
        )
        embeddings.extend([embedding.embedding for embedding in response.data])
    return embeddings

def compute_similarities(df1_tweets, df2_tweets):
    """计算 df1 中所有 tweet 与 df2 中所有 tweet 之间的相似度"""
    embeddings_df1 = get_embeddings(df1_tweets.tolist())
    embeddings_df2 = get_embeddings(df2_tweets.tolist())
    similarities = cosine_similarity(embeddings_df1, embeddings_df2)
    return similarities

def compute_similarities(df1_tweets, df2_tweets):
    """计算 df1 中所有 tweet 与 df2 中所有 tweet 之间的相似度"""
    embeddings_df1 = get_embeddings(df1_tweets.tolist())
    embeddings_df2 = get_embeddings(df2_tweets.tolist())
    similarities = cosine_similarity(embeddings_df1, embeddings_df2)
    return similarities

if __name__ == '__main__':
    for dataset in ['ArgMin', 'SemEval2016', 'VAST', 'covid-19-tweets','NLPCC2016', 'IBM-CSD', 'imgArg', 'PStance-main','FNC-1','IAC','perspectrum','emergent','poldeb']: #'ArgMin', 'SemEval2016', 'VAST', 'covid-19-tweets', 'IBM-CSD', 'imgArg', 'PStance-main','FNC-1','IAC','perspectrum','emergent','poldeb','perspectrum','emergent','poldeb',
        print("-----------------------{}-----------------------".format(dataset))
        # 加载数据
        test_df = pd.read_csv('./dataset/{}/split/test.csv'.format(dataset), index_col=None)
        train_df = pd.read_csv('./dataset/{}/split/train.csv'.format(dataset), index_col=None)
        if dataset !='NLPCC2016':
            for df in [test_df, train_df]:
                df.dropna(axis=0, how="any", inplace=True)
                df.reset_index(drop=True, inplace=True)
                df["Sentence"] = df["Sentence"].apply(lambda x: preprocessor.clean(x))
        # 计算全局相似度矩阵
        similarities = compute_similarities(test_df['Sentence'], train_df['Sentence'])
        top_n = 10
        test_df['top10_global_indices'] = [
            sorted(range(len(similarities[i])), key=lambda x: similarities[i][x], reverse=True)[:top_n]
            for i in range(len(test_df))
        ]
        n_train = len(train_df)
        test_df["random_indices"] = [
            np.random.choice(n_train, size=top_n, replace=False).tolist()
            for _ in range(len(test_df))
        ]
        # 保存结果
        test_df.to_csv('./dataset/{}/split/test_with_in_domain.csv'.format(dataset), index=False)