from sklearn.cluster import MiniBatchKMeans
from imageProcess import imageProcess
import numpy as np
import matplotlib.pyplot as plt


class KMeans_SIFT:
    def __init__(self,k):
        self.k = k

    def kmeans_process(self,matrix_image):
        batch_size = 20 * 3
        kmeans = KMeans(n_clusters=self.k, verbose=0).fit(matrix_image)
        return kmeans

    def newMatrixSift(self,data, kmeans, model):
        kmeans.verbose = False
        histo_list = []
        for des in data:
            # print(des)
            kp = np.asarray(des[1])
            # print (kp.shape)
            histo = np.zeros(self.k)
            nkp = np.size(kp)
            # print(histo)
            # print(nkp)
            for d in kp:
                    idx = kmeans.predict([d])
                    histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

            histo_list.append(histo)
        # print(np.asarray(histo_list).shape)
        path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Models'  +os.sep)
        with open(path + os.sep  + model +'.joblib', 'wb') as f1:
            joblib.dump(kmeans, f1)
        return np.asarray(histo_list)


imageDB = imageProcess('/home/anhnguyen/ASU/CSE-515/Project/Phase 2/Project - Phase 2/Data/Dataset2/')
imgs = imageDB.dbProcess(password='abcdefgh', process='f', model='s', dbase = 'imagedata_s')
# imgs_data = np.asarray(imgs)
imgs_data = []

i = -1
while i < len(imgs)-1:
    # print (x[1].shape)
    i += 1
    imgs_data.extend(imgs[i][1])

imgs_data = np.asarray(imgs_data)

dist = []
for i in range(100,1001,100):
    print(i)
    batch_size = 100
    kmeans = MiniBatchKMeans(batch_size =  batch_size, n_clusters=i)
    kmeans.fit(imgs_data)
    dist.append(kmeans.inertia_)
plt.title("Kmean Error with Clusters")
plt.plot(list(range(100,1001,100)), dist)
plt.show()