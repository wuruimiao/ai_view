from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


# df = pd.read_csv('mushrooms.csv')
# df.head()
#
# X = df.drop('class', axis=1)
# y = df['class']
# y = y.map({'p': 'Posionous', 'e': 'Edible'})
#
# cat_cols = X.select_dtypes(include='object').columns.tolist()
# for col in cat_cols:
#     print(f"col name : {col}, N Unique : {X[col].nunique()}")
#
#
# X_std = StandardScaler().fit_transform(X)
#
# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X_std)
# X_tsne_data = np.vstack((X_tsne.T, y)).T
# df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class '])
# df_tsne.head()
#
# plt.figure(figsize=(8, 8))
# sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
# plt.show()


def plot_embedding(data, label, train_index, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if i not in train_index:
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=plt.cm.Set1(label[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        else:
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=plt.cm.Set3(label[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # pdist = torch.nn.PairwiseDistance()
    # data = torch.tensor([[0.1, 0.1, 0.3],
    #                      [0.5, 0.6, 0.5],
    #                      [0.6, 0.7, 0.8]])
    result = []
    for i in range(100):
        data = torch.load(f"./demo/{i}_lat.pt", map_location=torch.device("cpu"))
        # print(data)
        data = data.cpu().detach().numpy()
        print(data)
        result.append(data)

    data = np.array(result)

    label = torch.tensor([0, 0, 1])
    label = label.numpy()
    train_index = [1]
    n_samples = data.shape[0]
    n_features = data.shape[1]
    print(data, label, n_samples, n_features)

    tsne = TSNE(n_components=3, init='pca', random_state=0)
    result = tsne.fit_transform(data)

    scatter = plt.scatter(result[:, 0], result[:, 1], c=y)
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, labels)
    # fig = plot_embedding(result[:,0:2], label,'t-SNE embedding of the data')
    # plot_embedding(result, label, train_index, 't-SNE embedding of the data')
#