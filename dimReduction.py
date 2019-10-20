from PostgresDB import PostgresDB
from imageProcess import imageProcess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.sparse.linalg import svds
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
# from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
import joblib
from sklearn.cluster import KMeans
import os
import math
import cv2


no_clusters = 400

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
      

class dimReduction(imageProcess):
    def __init__(self, dirpath, ext='*.jpg'):
        super().__init__(dirpath=dirpath, ext=ext)

    def fetchImagesAsPix(self, filename):
        image = cv2.imread(filename)
        size = np.asarray(image).shape
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        return img_yuv, size
    
    
    # Function to fetch Reduced dimensions from image
    def nmf(self, imageset, k, model_technique):
        model = NMF(n_components=k, init='random', random_state=0)
        scaler = StandardScaler(with_mean=False, with_std=True).fit(imageset)
        imageset= scaler.transform(imageset)
        W = model.fit_transform(imageset)
        H = model.components_
        path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Models'  +os.sep)

        with open(path + os.sep  + model_technique +'.joblib', 'wb') as f1:
            joblib.dump(model, f1)

        return W, H
    
    # Function to fetch Reduced dimensions from image
    def lda(self, imageset, k, model_technique):
        model = LatentDirichletAllocation(n_components=k, random_state=0)
        scaler = StandardScaler(with_mean=False, with_std=True).fit(imageset)
        imageset = scaler.transform(imageset)
        W = model.fit_transform(imageset)
        H = model.components_
        path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Models'  +os.sep)
        with open(path + os.sep  + model_technique +'.joblib', 'wb') as f1:
            joblib.dump(model, f1)

        return W, H
    
    
    # Function to perform PCA
    def pca(self, imageset, k, model):
        pca = PCA(n_components=k)
        data = pca.fit_transform(imageset)
        Sigma = np.diag(pca.explained_variance_)
        path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Models'  +os.sep)
        

        with open(path + os.sep  + model +'.joblib', 'wb') as f1:
            joblib.dump(pca, f1)

        return data, np.dot(data,np.linalg.inv(Sigma)), pca.components_
        # return u1, v2

    def svd(self,image, k, model):
        # U, s, Vt = svds(image, k)
        svd = TruncatedSVD(n_components=k)
        data = svd.fit_transform(image)
        # print(s.shape)
        # Sigma = np.zeros((image.shape[0], image.shape[1]))
        Sigma = np.diag(svd.singular_values_)
        # image = U[:,:self.k].dot(Sigma[:self.k, :self.k]).dot(V[:self.k,:])
        # print(image.shape)
        path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Models'  +os.sep)
        
        with open(path + os.sep  + model +'.joblib', 'wb') as f1:
            joblib.dump(svd, f1)
        return data, np.dot(data,np.linalg.inv(Sigma)) , svd.components_


    # Function to convert the List into string to insert into database
    def convString(self, lst):
        values_st = str(lst).replace('[', '\'{')
        values_st = values_st.replace(']', '}\'')
        return values_st

    # Method to get the sorted list of image contributions to the Basis Vectors
    def imgSort(self, h, imgs_meta):
        h_sort = [np.argsort(x)[::-1] for x in h]
        # print(imgs_meta)
        # print(h_sort)
        # print(np.asarray(imgs_meta).shape)
        # print(np.asarray(h_sort).shape)
        img_sort = []
        for idx, hs in enumerate(h_sort):
            img_sort.append([(imgs_meta[x], h[idx][x]) for x in hs])
        return img_sort

    # Method to get the sorted list of image contributions to features
    def imgFeatureSort(self, u, imgs):
        targ_imgs = []
        for vec in u:
            x = [(np.dot(vec, img), id) for id, img in imgs]
            y = sorted(x, key=lambda z: z[0], reverse=True)
            targ_imgs.append(y[0])

        return targ_imgs

    def ImgViz(self, images, savepath):
        no_images = len(images)
        columns = 3
        rows = no_images // columns
        if no_images % columns != 0:
            rows += 1
        ax = []
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle('Similar Images for Image {x}'.format(x=images[0]))
        plt.axis('off')
        for idx, i in enumerate(images):
            img = mpimg.imread(self.dirpath + self.ext.replace('*', i))
            ax.append(fig.add_subplot(rows, columns, idx + 1))
            if idx == 0:
                ax[-1].set_title("Original Image: " + i)  # set title
            else:
                ax[-1].set_title("Similar Image " + str(idx) + ":" + i)  # set title
            ax[-1].axis('off')
            plt.imshow(img)
        plt.savefig(savepath)
        plt.show()

    # Create table and insert data into it
    def createInsertDB(self, dbname, imgs_red, conn):
        cur = conn.cursor()
        # Create the table
        sql = "CREATE TABLE IF NOT EXISTS {db} (imageid TEXT NOT NULL, imagedata TEXT, PRIMARY KEY (imageid))".format(db=dbname)
        cur.execute(sql)
        conn.commit()

        for image in imgs_red:
            # print(image)
            sql = "SELECT {field} FROM {db} WHERE {field} = '{condition}';".format(field="imageid",db=dbname,condition=image[0])
            # print("SQL Check Exist - HOG: ", sql)
            cur.execute(sql)

            # cur.execute(sql)
            insert_value = str(np.asarray(image[1]).tolist())
            if cur.fetchone() is None:
                # print("Insert")
                # print("Not Exist HOG - Insert")
                sql2 = "INSERT INTO {db} VALUES('{x}', '{y}');".format(x=image[0],y=insert_value, db=dbname)
            else:
                # print("Update")
                # print("Exist HOG - Update")
                # column = "HOG"
                
                sql2 = "UPDATE {db} SET {x} ='{y}' WHERE IMAGEID = '{z}'".format(x="imagedata",y=insert_value,z = image[0], db=dbname)
            # Insert Values into the created table
            # sql2 = "INSERT INTO {db} VALUES {x}".format(db=dbname, x=imgs_red[2:-2])
            cur.execute(sql2)
        conn.commit()
        cur.close()
        print('Reduced Features saved successfully to Table {x}'.format(x=dbname))

    def simMetric(self, d1, d2):
        return 1 / (1 + self.l2Dist(d1, d2))

    # Function to create subject id matrix
    def subMatrix(self, conn, dbname, mat=True):
        # Read from the database and join with Meta data
        cur = conn.cursor()
        sqlj = "SELECT t2.subjectid, ARRAY_AGG(t1.imageid), ARRAY_AGG(t1.imagedata) FROM {db} " \
               "t1 INNER JOIN img_meta t2 ON t1.imageid = t2.image_id GROUP BY t2.subjectid".format(db=dbname)
        cur.execute(sqlj)
        subjects = cur.fetchall()
        sub_dict = {x: np.mean(np.array(y,dtype=float), axis=0) for x,z,y in subjects}
        sub_sim = {x:'' for x in sub_dict.keys()}
        sub_mat = []
        for x in sub_dict.keys():
            sub_sim[x] = sorted([(el, self.simMetric(sub_dict[x], sub_dict[el])) for el in sub_dict.keys() if el != x], key=lambda x:x[1], reverse=True)[0:3]
            sub_mat.append([self.simMetric(sub_dict[x], sub_dict[el]) for el in sub_dict.keys()])

        if mat == False:
            return sub_sim
        else:
            k = input('Please provide the number of latent semantics(k): ')
            w, h = self.nmf(np.array(sub_mat), int(k))
            img_sort = self.imgSort(h, list(sub_dict.keys()))
        return np.array(img_sort)

    def binMat(self, conn, dbname):
        # Read from the database and join with Meta data
        cur = conn.cursor()
        sqlj = "SELECT t1.imageid, CASE WHEN t2.orient = 'left' THEN 1 ELSE 0 END , " \
               "CASE WHEN t2.orient = 'right' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.aspect = 'dorsal' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.aspect = 'palmar' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.accessories = '1' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.accessories = '0' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.gender = 'male' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.gender = 'female' THEN 1 ELSE 0 END" \
               " FROM {db} " \
               "t1 INNER JOIN img_meta t2 ON t1.imageid = t2.image_id".format(db=dbname)
        cur.execute(sqlj)
        subjects = cur.fetchall()
        img_meta = []
        bin_mat = []
        for x in subjects:
            img_meta.append(x[0])
            bin_mat.append(x[1:])
        k = input('Please provide the number of latent semantics(k): ')
        w, h = self.nmf(np.array(bin_mat).T, int(k))
        img_sort = self.imgSort(h, img_meta)
        features = ['left', 'right', 'dorsal', 'palmar', 'acessories', 'no_accessories', 'male', 'female']
        feature_sort = [np.argsort(x)[::-1] for x in w.T]
        feat_ls = []
        for idx, x in enumerate(feature_sort):
            feat_ls.append([(features[i], w.T[idx][i]) for i in x])
        return img_sort, feat_ls

      
    def normalize(self, imgs):
        print(imgs)
        temp = np.array([[1/math.exp(t) for t in x] for x in imgs])
        return temp
    def hist(self, imgs):
        mean = np.array([[x[t] for t in range(3)] for x in imgs])
        sd = np.array([[x[t] for t in range(3, 6)] for x in imgs])
        sk = np.array([[x[t] for t in range(6, 10)] for x in imgs])
        (m_histogram, m_bin_edges) = np.histogram(mean.ravel(), bins=10)
        (sd_histogram, sd_bin_edges) = np.histogram(sd.ravel(), bins=10)
        (sk_histogram, sk_bin_edges) = np.histogram(sk.ravel(), bins=10)
        return np.array([np.array(m_histogram), np.array(sd_histogram), np.array(sk_histogram)])

    
    def singleImageFetch(self, img, feature):
        filename = self.dirpath + img + '.jpg'
        print(filename)
        if feature == 'm':
            pixels, size = self.fetchImagesAsPix(filename)
            vals = self.imageMoments(pixels, size)
        elif feature == 's':
            vals = self.sift_features(filename)
        elif feature == 'h':
            vals = self.hog_process(filename)
        elif feature == 'l':
            vals = self.lbp_preprocess(filename)
        else:
            print('Incorrect value for Model provided')
            exit()
        return vals
    
    
    # Classify images based on label
    def classifyImg(self, conn, feature, img, label, dim):
                # fetch image dataset
        db_feature = 'imagedata_' + feature + '_' + dim + '_' + label

        cur = conn.cursor()
        sqlj = "SELECT imageid, imagedata FROM {db}".format(db=db_feature)
        cur.execute(sqlj)
        label_data = cur.fetchall()
        recs_flt = []
        img_meta = []
        for rec in label_data:
            recs_flt.append(eval(rec[1]))
            img_meta.append(rec[0])

        image = self.singleImageFetch(img=img, feature=feature)
        if feature == 'm':
            image = [x for y in image for x in y]

        # Check for which reduced dimension technique is being used
        path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Models'  +os.sep)
        # with open(path + os.sep  + model +'.joblib', 'wb')
        # path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'Phase1\\Models' + os.sep)
        model = joblib.load(path + os.sep + "{0}_{1}_{2}.joblib".format(feature, dim, label))
        imgs_red = np.array(recs_flt)
        clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
        clf.fit(imgs_red)
        image = np.array(image)
        image = np.dot(image, model.components_.T)        
        # image = model.transform(image.reshape(1,-1))
        pred = clf.predict(image.reshape(1,-1))
        x = clf.decision_function(image.reshape(1,-1))
        print(x)
        print(pred)

    # Function to save the reduced dimensions to database

    def saveDim(self, feature, model, dbase, k, password='1Idontunderstand',
                host='localhost', database='postgres',
                user='postgres', port=5432, label=None, meta=False, negative_handle ='n'):

        imageDB = imageProcess(self.dirpath)
        imgs = imageDB.dbProcess(password=password, process='f', model=feature, dbase = dbase)
        kmeans_model = 'kmeans_' + str(no_clusters)
        technique_model = feature + '_' + model
        
        
        if label is not None:
            filteredImage = imageDB.CSV(label)
            label = label.replace(" ", "_")
            dbase += '_' + model + '_' + label
            kmeans_model += '_' + label
            technique_model += '_' + label
        else:
            dbase += '_' + model

        # print(technique_model)
        imgs_data = []
        imgs_meta = []

        i = -1
        while i < len(imgs)-1:
            # print (x[1].shape)
            i += 1
            if label is not None and imgs[i][0] not in filteredImage:
                # print("label")
                del imgs[i]
                i -= 1
                continue
            if feature == "s" or (feature == "m" and model in ("nmf", "lda")):
                imgs_data.extend(imgs[i][1])
            else:
                imgs_data.append(imgs[i][1].reshape((-1)))  
                    # print (image_cmp.shape)
            imgs_meta.append(imgs[i][0])
            # print(i)
            # print(len(imgs))
        
        #Handle Negative Value of NMF
        # if feature == 'm' and (model == 'lda' or model == 'nmf'):
        #     print ("Normalize")
        #     if negative_handle == 'h':
        #         imgs_data = self.hist(imgs_data)
        #     else:
        #         imgs_data = self.normalize(imgs_data)
        
        imgs_data = np.asarray(imgs_data)
        # print(imgs_data.shape)
        # print(imgs_meta)
        # imgs_meta = [x[0] if x[0] in filteredImage for x in imgs]
        imgs_zip = list(zip(imgs_meta, imgs_data))
        db = PostgresDB(password=password, host=host,
                        database=database, user=user, port=port)
        conn = db.connect()
        if meta:
            imageDB.createInsertMeta(conn)

        model = model.lower()

        if feature == "s" or (feature == "m" and model in ("nmf", "lda")):
            if imgs_data.shape[0] < no_clusters:
                Kmeans = KMeans_SIFT(imgs_data.shape[0] // 2)
            else:
                Kmeans = KMeans_SIFT(no_clusters)
            clusters = Kmeans.kmeans_process(imgs_data)
            # print (imgs_zip)
            imgs_data = Kmeans.newMatrixSift(imgs, clusters ,kmeans_model)
            imgs_zip = list(zip(imgs_meta, imgs_data))

        if model == 'nmf':
            w, h = self.nmf(imgs_data, k, technique_model)
            imgs_red = np.dot(imgs_data, h.T).tolist()
            # print(np.asarray(w).shape)
            # print(np.asarray(h).shape)
            imgs_sort = self.imgSort(w.T, imgs_meta)
            feature_sort = self.imgFeatureSort(h, imgs_zip)

        elif model == 'lda':
            w, h = self.lda(imgs_data, k, technique_model)
            imgs_red = np.dot(imgs_data, h.T).tolist()
            # print(np.asarray(w).shape)
            # print(np.asarray(h).shape)
            imgs_sort = self.imgSort(w.T, imgs_meta)
            feature_sort = self.imgFeatureSort(h, imgs_zip)

        elif model == 'pca':
            data, U, Vt = self.pca(imgs_data, k, technique_model)
            imgs_red = data.tolist()
            imgs_sort = self.imgSort(U.T, imgs_meta)
            feature_sort = self.imgFeatureSort(Vt, imgs_zip)

        elif model == 'svd':
            # print(imgs_data.shape)
            data, U, Vt = self.svd(imgs_data, k, technique_model)
            imgs_red = data.tolist()
            # print(im)
            # U[:,:self.k].dot(Sigma[:self.k, :self.k]).dot(V[:self.k,:])
            imgs_sort = self.imgSort(U.T, imgs_meta)
            feature_sort = self.imgFeatureSort(Vt, imgs_zip)

        # print("=======================")
        # print(imgs_sort)
        # print("=======================")
        # print(feature_sort)
        # Process the reduced Images
        imgs_red = list(zip(imgs_meta, imgs_red))
        # print (np.asarray(imgs_sort).shape)
        # print(img_sort)
        # print (np.asarray(feature_sort).shape)
        self.createInsertDB(dbase, imgs_red, conn)
        return imgs_sort, feature_sort
