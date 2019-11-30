from gensim.models import Word2Vec
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time


save_path = r'D:\code\github\zxx\data\word2vec.bin'
def train_and_save():
    data_file = "../../data/all_tweets_lda_5.jl"

    sentences = []
    with open(data_file, encoding='utf-8') as fin:
        for line in fin.readlines():
            data = json.loads(line)
            sentences.append(data['cleaned'].split(' '))
    print('begin to train')
    model = Word2Vec(sentences, size=60)
    print('finish trainning')

    show_words = [('trump', 'chinese'), ('trump', 'u'), ('market', 'economy')]
    for words in show_words:
        print(model.wv.similarity(words[0], words[1]))
    model.save(fname_or_handle=save_path)


def plot_pca_tsen(X, y):
    feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    print('Size of the dataframe: {}'.format(df.shape))

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 3),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df["pca-one"],
        ys=df["pca-two"],
        zs=df["pca-three"],
        c=df["y"],
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 3),
    #     data=df,
    #     legend="full",
    #     alpha=0.3
    # )

    plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 3),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax1
    )
    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 3),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax2
    )
    plt.show()


def _to_vec(model, stc):
    words = stc.split(' ')
    return np.mean([model.wv[w] for w in words if w in model.wv], axis=0)


def plot_label():
    label_file = r'D:\code\github\zxx\data\选择出来的标注数据.xlsx'
    df_label = pd.read_excel(label_file)
    model = Word2Vec.load(save_path)
    df_label['vec'] = df_label['cleaned'].apply(lambda x: _to_vec(model, x))
    X = np.asarray([vec for vec in df_label['vec'].values])

    map1={'CENTRAL':2, 'POSITIVE':0, 'NEGATIVE':1}
    df_label['label'] = df_label['label'].apply(lambda x: map1[x])
    y = np.asarray(df_label['label'])
    print(X.shape, y.shape)
    plot_pca_tsen(X, y)


def infer():
    lda5_file = r'D:\code\github\zxx\data\all_tweets_lda_5.jl'
    senti_file = r'D:\code\github\zxx\data\all_tweets_senti.jl'
    label_file = r'D:\code\github\zxx\data\选择出来的标注数据.xlsx'
    df_lda5 = pd.read_json(lda5_file, orient='records', lines=True)
    df_lda5['datetime'] = df_lda5['datetime'].astype(str)
    df_label = pd.read_excel(label_file)
    model = Word2Vec.load(save_path)
    vectors = np.asarray([_to_vec(model, tweet) for tweet in df_label['cleaned'].values])
    norm_vectors = np.linalg.norm(vectors, axis=1)
    print(vectors.shape, norm_vectors.shape)
    labels = [v for v in df_label['label'].values]
    def _infer(tweet):
        vec = _to_vec(model, tweet)
        tmp = np.dot(vectors, vec)
        tmp = tmp / (norm_vectors * np.linalg.norm(vec))
        _idx = np.argmax(tmp)
        return labels[_idx]
    df_lda5['senti'] = df_lda5['cleaned'].apply(_infer)

    for senti in ['POSITIVE', 'NEGATIVE', 'CENTRAL']:
        print(senti, sum((df_lda5['senti'] == senti).astype(int)))

    df_lda5.to_json(senti_file, orient='records', lines=True)



if __name__ == '__main__':
    # train_and_save()
    # plot_label()
    infer()