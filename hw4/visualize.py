import numpy as np
from skimage import data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

# def PCAdecompose(images):
#     images = np.array(images)
#     print('original shape:', images.shape)
#     mean = np.mean(images, axis=0)
#     images = images - mean
#     u, w, v = np.linalg.svd(images, full_matrices=False)
#     summation = np.sum(w)
#     w = w * 100 / summation
    
#     eigen = 0;
#     for i in range(300):
#         eigen = eigen + round(w[i], 1)
#         print("num#%i : %i " % (i, eigen))


fileName = 'data/visualization.npy'
array = np.load(fileName)
# PCAdecompose(array)



pca = PCA(n_components=30, copy=False, whiten=True, svd_solver='full')
newData = pca.fit_transform(array)
print('reduced dimension ', newData.shape)

tsne = TSNE(n_components=2, verbose=1, n_iter=300, random_state=100)
newData = tsne.fit_transform(newData)
print(newData.shape)
kmeans = KMeans(n_clusters=2, random_state=100).fit(newData)
Label = kmeans.labels_

zero_c = 0
one_c = 0
for i in range(5000):
    if Label[i] == 1:
        one_c = one_c + 1
    else:
        zero_c = zero_c + 1

print(zero_c, one_c)

zero_c = 0
one_c = 0
for i in range(5000,10000,1):
    if Label[i] == 1:
        one_c = one_c + 1
    else:
        zero_c = zero_c + 1

print(zero_c, one_c)

realLabel = np.ones(10000)
for i in range(5000):
    realLabel[i] = 0

plt.scatter(newData[:, 0], newData[:, 1], c=Label, s=5, cmap='coolwarm')
plt.xlabel('tsne x')
plt.ylabel('tsne y')
plt.title('tsne dimension reduction (predict label)')
plt.show()
