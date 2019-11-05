from PostgresDB import PostgresDB
from imageProcess import imageProcess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
import joblib
import os
import math
import cv2
import matplotlib.gridspec as gs
import tqdm


no_clusters = 400

class KMeans_SIFT(imageProcess):
    def __init__(self,k, ext='*.jpg'):
        self.k = k
        super().__init__(ext=ext)

    def kmeans_process(self,matrix_image):
        batch_size = 800
        kmeans = MiniBatchKMeans(batch_size=batch_size, n_clusters=self.k, verbose=0).fit(matrix_image)
        return kmeans

    def newMatrixSift(self,data, kmeans, model):
        kmeans.verbose = False
        histo_list = []
        for des in data:
            kp = np.asarray(des[1])
            histo = np.zeros(self.k)
            nkp = kp.shape[0]
            for d in kp:
                    idx = kmeans.predict([d])
                    histo[idx] += 1/nkp
            histo_list.append(histo)

        path = self.modelpath
        with open(path + os.sep  + model +'.joblib', 'wb') as f1:
            joblib.dump(kmeans, f1)
        return np.asarray(histo_list)
      

class dimReduction(imageProcess):
    def __init__(self, ext='*.jpg'):
        super().__init__(ext=ext)

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
        path = self.modelpath

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
        path = self.modelpath
        with open(path + os.sep  + model_technique +'.joblib', 'wb') as f1:
            joblib.dump(model, f1)

        return W, H
    
    
    # Function to perform PCA
    def pca(self, imageset, k, model):
        pca = PCA(n_components=k)
        data = pca.fit_transform(imageset)
        Sigma = np.diag(pca.explained_variance_)
        path = self.modelpath
        

        with open(path + os.sep  + model +'.joblib', 'wb') as f1:
            joblib.dump(pca, f1)

        return data, np.dot(data,np.linalg.inv(Sigma)), pca.components_
        # return u1, v2

    def svd(self,image, k, model):
        svd = TruncatedSVD(n_components=k)
        data = svd.fit_transform(image)
        Sigma = np.diag(svd.singular_values_)
        path = self.modelpath
        
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

    def imgViz2(self, images, savepath = ''):
        if not savepath:
            savepath = self.dirpath
        no_images = 5
        column = 5
        rows = len(images)
        fig = plt.figure(figsize=(rows*10, 20))
        spec = gs.GridSpec(ncols=column, nrows=rows, figure=fig)
        plt.axis('off')
        for idx, i in enumerate(images):
            for imag in range(no_images):
                img = mpimg.imread(savepath + os.sep + i[imag][0] + '.jpg')
                ax = fig.add_subplot(spec[idx, imag])

                ax.set_title(i[imag][0] + '\nScore: ' + str(float(i[imag][1])))  # set title
                ax.axis('off')
                ax.imshow(img)
        plt.savefig(savepath+ os.sep +'Output')
        plt.show()

    # Visualize image feature latent semantics
    def imgViz_feature2(self, images, dirpath):
        column = 5
        rows = int(column / 5) + 1
        fig = plt.figure(figsize=(30, 40))
        spec = gs.GridSpec(ncols=column, nrows=rows, figure=fig)
        plt.axis('off')
        for idx, i in enumerate(images):
            img = mpimg.imread(dirpath + os.sep + i[1] + '.jpg')
            ax = fig.add_subplot(spec[0, idx])
            ax.set_title(i[1] + '\nScore: ' + str(float(i[0])))  # set title
            ax.axis('off')
            ax.imshow(img)
        plt.savefig(dirpath+os.sep+'Output')
        plt.show()

    
    def imgViz(self, images, savepath = ''):
        if not savepath:
            savepath = self.dirpath
        no_images = 10
        column = 5
        rows = len(images)
        for idx, i in enumerate(images):
            fig = plt.figure(figsize=(40, 30))
            fig.suptitle('Latent Semantic: ' + str(idx+1), fontsize=60)
            plt.axis('off')
            spec = gs.GridSpec(ncols=column, nrows=2, figure=fig)
            for imag in range(no_images):
                img = mpimg.imread(savepath + i[imag][0] + '.jpg')
                if imag < 5:
                    ax = fig.add_subplot(spec[0, imag])
                else:
                    pc = imag - 5
                    ax = fig.add_subplot(spec[1, pc])

                ax.set_title(i[imag][0] + '\nScore: ' + str(float(i[imag][1])), fontsize=25)  # set title
                ax.axis('off')
                ax.imshow(img)
            if not os.path.exists(savepath+'Output'):
                os.makedirs(savepath+'Output')
            plt.savefig(savepath +'Output' + os.sep + 'DLS-' +str(idx + 1))

    # Visualize image feature latent semantics
    def imgViz_feature(self, images, dirpath = ''):
        if not dirpath:
            dirpath = self.dirpath
        column = 5
        rows = 2 if len(images) > 5 else 1
        p=0
        pidx = 0
        print("Saving images each latent semantic at a time")
        pc = 0
        for idx, i in enumerate(images):
            if idx % 10 == 0:
                if idx / 10 != 0:
                    pc = pc + 1
                    if not os.path.exists(dirpath + 'Output'):
                        os.makedirs(dirpath + 'Output')
                    plt.savefig(dirpath + 'Output' + os.sep + 'FLS' + str(pc))
                fig = plt.figure(figsize=(60, 40))
                spec = gs.GridSpec(ncols=column, nrows=rows, figure=fig)
                plt.axis('off')
                p = 0
            img = mpimg.imread(dirpath + i[1] + '.jpg')
            ax = fig.add_subplot(spec[p, pidx])
            ax.set_title(i[1] + '\nScore: ' + str(float(i[0])), fontsize=30)  # set title
            ax.axis('off')
            ax.imshow(img)
            pidx += 1
            if (idx+1) % 5 == 0:
                p += 1
                pidx = 0
        if not os.path.exists(dirpath + 'Output'):
            os.makedirs(dirpath + 'Output')
        plt.savefig(dirpath + 'Output' + os.sep + 'FLS' + str(pc + 1))
        # plt.show()
    
    
    # Create table and insert data into it
    def createInsertDB(self, dbname, imgs_red, conn):
        cur = conn.cursor()
        # Create the table
        sql = "CREATE TABLE IF NOT EXISTS {db} (imageid TEXT NOT NULL, imagedata TEXT, PRIMARY KEY (imageid))".format(db=dbname)
        cur.execute(sql)
        conn.commit()

        for image in imgs_red:
            sql = "SELECT {field} FROM {db} WHERE {field} = '{condition}';".format(field="imageid",db=dbname,condition=image[0])
            cur.execute(sql)
            insert_value = str(np.asarray(image[1]).tolist())
            if cur.fetchone() is None:
                sql2 = "INSERT INTO {db} VALUES('{x}', '{y}');".format(x=image[0],y=insert_value, db=dbname)
            else:
                sql2 = "UPDATE {db} SET {x} ='{y}' WHERE IMAGEID = '{z}'".format(x="imagedata",y=insert_value,z = image[0], db=dbname)
            cur.execute(sql2)
        conn.commit()
        cur.close()
        print('Reduced Features saved successfully to Table {x}'.format(x=dbname))

    # Similarity Metric
    def simMetric(self, d1, d2):
        return 1 / (1 + cv2.norm(d1, d2, cv2.NORM_L2))

    # Function to create subject id matrix
    def subMatrix(self, conn, dbname, subject="", mat=True):
        # Read from the database and join with Meta data
        cur = conn.cursor()
        sqlj = "SELECT t2.subjectid, ARRAY_AGG(t1.imageid), ARRAY_AGG(t1.imagedata) FROM {db} " \
               "t1 INNER JOIN img_meta t2 ON t1.imageid = t2.image_id GROUP BY t2.subjectid".format(db=dbname)
        cur.execute(sqlj)
        subjects = cur.fetchall()
        sub_dict = {x: np.mean(np.array([eval(p) for p in y], dtype=float), axis=0) for x, z, y in subjects}
        if subject != "":
            if subject in [x for x,y,z in subjects]:
                sub_simd = {x: '' for x in sub_dict.keys()}
                for x in sub_dict.keys():
                    sub_simd[x] = sorted([(el, self.simMetric(sub_dict[x], sub_dict[el])) for el in sub_dict.keys() if el != x], key=lambda x:x[1], reverse=True)[0:3]
                sub_sim = sub_simd[subject]
            else:
                print('Provided Subject ID not found in small dataset! Searching 11K Images...')
                sqlm = "SELECT image_id FROM img_meta WHERE subjectid = '{s}'".format(s=subject)
                cur.execute(sqlm)
                sub_imgs = cur.fetchall()
                sub_features = self.subImageFetch(set([x[0] for x in sub_imgs]))
                path = self.modelpath
                model = joblib.load(path + os.sep + "l_svd.joblib")
                sub_dim = np.dot(np.array(sub_features), model.components_.T)
                sub_cent = np.mean(sub_dim, axis=0)
                sub_sim = sorted([(el, self.simMetric(sub_cent, sub_dict[el])) for el in sub_dict.keys()],
                                 key=lambda x: x[1], reverse=True)[0:3]
        conn.close()
        sub_mat = []
        if mat == False:
            return sub_sim
        else:
            for x in sub_dict.keys():
                sub_mat.append([self.simMetric(sub_dict[x], sub_dict[el]) for el in sub_dict.keys()])
            k = input('Please provide the number of latent semantics(k): ')
            w, h = self.nmf(np.array(sub_mat), int(k), model_technique='')
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
        w, h = self.nmf(np.array(bin_mat).T, int(k), model_technique='')
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

    
    # Sub Image Fetch
    def subImageFetch(self, images):
        path = self.ogpath
        ext = '.jpg'
        vals = []
        pbar = tqdm.tqdm(total=len(images))
        for im in images:
            filepath = path + im + ext
            val = self.lbp_preprocess(filepath)
            vals.append(val)
            pbar.update(1)
        return vals

    # Single Image Fetch
    def singleImageFetch(self, img, feature):
        filename = self.dirpath + img + '.jpg'
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
        label = label.replace(" ", "_")
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

        # Check for which reduced dimension technique is being used
        path = self.modelpath
        model = joblib.load(path + os.sep + "{0}_{1}_{2}.joblib".format(feature, dim, label))
        imgs_red = np.array(recs_flt)
        clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=1)
        clf.fit(imgs_red)
        image = np.asarray(image)
        # print(image.shape)
        if feature == 's' or (feature == 'm' and dim in ('nmf', 'lda')):
            kmeans = joblib.load(path + os.sep + 'kmeans_{0}_{1}_{2}.joblib'.format(model.components_.shape[1], feature, label))
            histo = np.zeros(model.components_.shape[1])
            nkp = image.shape[0]
            for d in image:
                idx = kmeans.predict([d])
                histo[idx] += 1/nkp
            image = np.asarray(histo)
        else:
            image = image.reshape((-1))
        if dim == 'pca':
            image = model.transform([image])
        else:
            image = np.dot(image, model.components_.T)
        pred = clf.predict(image.reshape(1,-1))
        return pred

    # Function to save the reduced dimensions to database

    def saveDim(self, feature, model, dbase, k, label=None, meta=False):

        imageDB = imageProcess(self.dirpath)
        imgs = imageDB.dbProcess(process='f', model=feature, dbase=dbase)
        kmeans_model = 'kmeans_' + str(no_clusters) + '_' + feature
        technique_model = feature + '_' + model

        db = PostgresDB()
        conn = db.connect()
        
        if meta:
            imageDB.createInsertMeta(conn)
        if label is not None:
            imgs = imageDB.CSV(conn, dbase, label)
            label = label.replace(" ", "_")
            dbase += '_' + model + '_' + label
            kmeans_model += '_' + label
            technique_model += '_' + label
        else:
            dbase += '_' + model

        imgs_data = []
        imgs_meta = []
        for img in imgs:
            if feature == "s" or (feature == "m" and model in ("nmf", "lda")):
                imgs_data.extend(img[1])
            else:
                imgs_data.append(img[1].reshape((-1)))

            imgs_meta.append(img[0])

        imgs_data = np.asarray(imgs_data)
        imgs_zip = list(zip(imgs_meta, imgs_data))
    
        model = model.lower()

        if feature == "s" or (feature == "m" and model in ("nmf", "lda")):
            if imgs_data.shape[0] < no_clusters:
                Kmeans = KMeans_SIFT(imgs_data.shape[0] // 2)
            else:
                Kmeans = KMeans_SIFT(no_clusters)
            clusters = Kmeans.kmeans_process(imgs_data)
            imgs_data = Kmeans.newMatrixSift(imgs, clusters ,kmeans_model)
            imgs_zip = list(zip(imgs_meta, imgs_data))

        if model == 'nmf':
            w, h = self.nmf(imgs_data, k, technique_model)
            imgs_red = np.dot(imgs_data, h.T).tolist()
            imgs_sort = self.imgSort(w.T, imgs_meta)
            feature_sort = self.imgFeatureSort(h, imgs_zip)
            U = w 
            Vt = h

        elif model == 'lda':
            w, h = self.lda(imgs_data, k, technique_model)
            imgs_red = np.dot(imgs_data, h.T).tolist()
            imgs_sort = self.imgSort(w.T, imgs_meta)
            feature_sort = self.imgFeatureSort(h, imgs_zip)
            U = w 
            Vt = h

        elif model == 'pca':
            data, U, Vt = self.pca(imgs_data, k, technique_model)
            imgs_red = data.tolist()
            imgs_sort = self.imgSort(U.T, imgs_meta)
            feature_sort = self.imgFeatureSort(Vt, imgs_zip)

        elif model == 'svd':
            data, U, Vt = self.svd(imgs_data, k, technique_model)
            imgs_red = data.tolist()
            imgs_sort = self.imgSort(U.T, imgs_meta)
            feature_sort = self.imgFeatureSort(Vt, imgs_zip)

        imgs_red = list(zip(imgs_meta, imgs_red))
        self.createInsertDB(dbase, imgs_red, conn)
        return imgs_sort, feature_sort, U, Vt
