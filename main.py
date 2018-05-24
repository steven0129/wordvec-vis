import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.cluster import KMeans

classes = np.load('classes.npy')
word2vec = np.load('word2vec.npy')

labels = []
tokens = []

print('輸入word vector...')
for index, label in enumerate(tqdm(classes)):
    labels.append(str(label))
    tokens.append(list(word2vec[index]))

print('輸入TSNE模型...')
tsneModel = TSNE(n_components=3, init='pca', n_iter_without_progress=2500, verbose=2)
newValues = tsneModel.fit_transform(tokens)

print('儲存可視化座標...')
XYZ = []

for index, value in tqdm(enumerate(newValues)):
    XYZ.append(value)

kmeans = KMeans(n_clusters=6, verbose=2).fit(XYZ)

with open('result.csv', 'w') as f:
    f.write('Label,X,Y,Z,Clsuter\n')
    for index, value in enumerate(kmeans.labels_):
        X = XYZ[index][0]
        Y = XYZ[index][1]
        Z = XYZ[index][2]

        f.write(f'{labels[index]},{X},{Y},{Z},{value}\n')


