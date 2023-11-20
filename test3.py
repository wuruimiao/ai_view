# imports from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.datasets import load_iris

# %matplotlib inline
# %pylab inline

from sklearn.manifold import TSNE
# from sklear.decomposition import PCA

iris = load_iris()
X_train = iris.data


#TSNE(2)
transformed = TSNE(n_components=2, random_state=0).fit_transform(X_train)

# x, y = list(zip(*X_2d))

plt.figure(figsize=(9, 6))
plt.scatter(transformed[:,0], transformed[:,1])
plt.legend()
plt.show()

# #TSNE(3)
# data = TSNE(n_components=3, random_state=0).fit_transform(X_train)
# x, y, z = list(zip(*data))
#
# fig = pylab.figure()
# ax = fig.add_subplot(111, projection = '3d')
# sc = ax.scatter(x,y,z)

# # PCA(2)
# x, y = list(zip(*MinMaxScaler().fit_transform(PCA(2).fit_transform(X_train[:]))))
#
# plt.figure(figsize=(9, 6))
# plt.scatter(x, y)
# plt.legend()
# plt.show()
#
# # PCA(3)
# x, y, z = list(zip(*MinMaxScaler().fit_transform(PCA(3).fit_transform(X_train[:]))))
#
# fig = pylab.figure()
# ax = fig.add_subplot(111, projection = '3d')
# sc = ax.scatter(x,y,z)