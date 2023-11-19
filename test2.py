import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import manifold

# load dataset
data = datasets.fetch_openml(
    'Fashion-MNIST',
    version=1,
    return_X_y=True
)
# data returns a tuple (features, target)
features, target = data
target = target.astype(int)

# reshape the features for plotting image
image = features.iloc[0].values.reshape(28, 28)
plt.imshow(image, cmap='gray')

# dimensionality reduction using t-SNE
tsne = manifold.TSNE(n_components=2, random_state=42)
# fit and transform
mnist_tr = tsne.fit_transform(features[:30000])
# transformed_data is a 2D numpy array of shape (30000, 2)


# create dataframe
cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
                      data=np.column_stack((mnist_tr,
                                            target.iloc[:30000])))
# cast targets column to int
cps_df.loc[:, 'target'] = cps_df.target.astype(int)
cps_df.head()

clothes_map = {0: 'T-shirt/top',
               1: 'Trouser',
               2: 'Pullover',
               3: 'Dress',
               4: 'Coat',
               5: 'Sandal',
               6: 'Shirt',
               7: 'Sneaker',
               8: 'Bag',
               9: 'Ankle Boot'}
# map targets to actual clothes for plotting
cps_df.loc[:, 'target'] = cps_df.target.map(clothes_map)

cps_df.target.value_counts().plot(kind='bar')

grid = sns.FacetGrid(cps_df, hue="target", size=8)
grid.map(plt.scatter, 'CP1', 'CP2').add_legend()
