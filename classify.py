from PostgresDB import PostgresDB
from dimReduction import dimReduction
from clustering import KMeans
import glob
import numpy as np
import os
class classify(dimReduction):
    def __init__(self, feature='h', dim='svd', k=20):
        super().__init__(ext='*.jpg')
        db = PostgresDB()
        self.conn = db.connect()
        self.table_f = 'imagedata' + '_' + feature
        self.table_d = 'imagedata' + '_' + feature + '_' + dim
        self.feature = feature
        self.dim = dim
        self.k = k

    # Check if tables exist and create otherwise
    def checkCreate(self):
        # Calculate feature Latent semantics if not exists
        cur = self.conn.cursor()

        try:
            lsf_m, lsf_d, lsf_f = (False,), (False,), (True,)
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = '{0}')".format(self.table_d))
            lsf_d = cur.fetchone()
            if not lsf_d[0]:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = '{0}')".format(self.table_f))
                lsf_f = cur.fetchone()
            cur.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'img_meta')")
            lsf_m = cur.fetchone()
            # Check if table exists if not create table
            if not lsf_f[0]:
                print('Feature Data Being Loaded!')
                self.dbProcess(model=self.feature, process='s')
            if not lsf_d[0]:
                print('Dimension Reduction Data Being Loaded!')
                _, _, _, _ = self.saveDim(self.feature, self.dim, self.table_f, self.k)
            if not lsf_m[0]:
                print('Meta Data being Loaded!')
                self.createInsertMeta(self.conn)
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(e)
            exit(-1)

    # Fetch data for a single aspect
    def fetchAspect(self, aspect):
        cur = self.conn.cursor()
        cur.execute("SELECT imageid, imagedata FROM {0} t1 INNER JOIN img_meta t2 "
                    "ON t1.imageid = t2.image_id AND t2.aspect = '{1}'".format(self.table_f, aspect))
        imgs = cur.fetchall()
        # Separate Image IDs and Image data
        ids = []
        data = []
        for img in imgs:
            ids.append(img[0])
            data.append(eval(img[1]))
        return ids, data

    # Fetch Test data and insert into database
    def fetchTestData(self):
        cur = self.conn.cursor()
        # Check if table already exists
        cur.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'imagedata_test')")
        tflg = cur.fetchone()
        if not tflg[0]:
            values = []
            names = []
            # Fetch images as pixels
            for filename in glob.glob(self.testpath + self.ext):
                h_val = self.hog_process(filename)
                values.append(str(np.asarray(h_val).tolist()))
                name = os.path.basename(filename)
                name = os.path.splitext(name)[0]
                names.append(name)
            values_zip = str(set(zip(names, values))).replace('{', ' ')
            values_zip = values_zip.replace('}', ' ')
            cur.execute("CREATE TABLE imagedata_test "
                        "AS WITH t(imageid, imagedata) AS (VALUES {0}) SELECT * FROM t".format(values_zip))

        # Read data from Table
        cur.execute("SELECT * FROM imagedata_test")
        data = cur.fetchall()
        test_data = []
        test_ids = []
        for d in data:
            test_ids.append(d[0])
            test_data.append(eval(d[1]))
        self.conn.commit()
        cur.close()
        return test_ids, test_data

    # Task 1 Function to perform LSA
    def LSAnalysis(self):
        # Check if tables exist otherwise create them
        self.checkCreate()
        # Now segregate the content between dorsal and palmar
        dorsal_ids, dorsal_data = self.fetchAspect('dorsal')
        palmar_ids, palmar_data = self.fetchAspect('palmar')

        # Perfrom SVD and get the Feature Latent Semantics for both dorsal and palmar
        _, _, lsDorsal = self.svd(dorsal_data, self.k, self.feature + '_' + self.dim + '_' + 'dorsal')
        _,_, lsPalmar = self.svd(palmar_data, self.k, self.feature + '_' + self.dim + '_' + 'palmar')

        # Fetch the test data set and transform them into feature space
        test_ids, test_data = self.fetchTestData()
        test_data = np.asarray(test_data)

        # Fetch the labels for the images
        cur = self.conn.cursor()
        test_ids_st = str(set(test_ids)).replace('{', '(')
        test_ids_st = test_ids_st.replace('}', ')')
        cur.execute("SELECT image_id, aspect FROM img_meta WHERE image_id IN {0}".format(test_ids_st))
        labels = cur.fetchall()
        label_dict = {el[0]:el[1] for el in labels}
        cnt = 0
        for idx, test in enumerate(test_data):
            dorsalValue = np.linalg.norm(np.dot(test, lsDorsal.T))
            palmarValue = np.linalg.norm(np.dot(test, lsPalmar.T))
            if dorsalValue > palmarValue:
                pred = 'dorsal'
            else:
                pred = 'palmar'
            if label_dict[test_ids[idx]] == pred:
                cnt = cnt + 1
            print(test_ids[idx])
            print(label_dict[test_ids[idx]])
            print('dorsal: ', dorsalValue)
            print('palmar: ', palmarValue)

        print('\nAccuracy: ', cnt/len(test_data))

    # Task 2 function to use clustering to classify dorsal palmar
    def clusterClassify(self, k=9):
        # Check if tables exist otherwise create them
        self.checkCreate()
        # Now segregate the content between dorsal and palmar
        dorsal_ids, dorsal_data = self.fetchAspect('dorsal')
        palmar_ids, palmar_data = self.fetchAspect('palmar')

        # Compute Cluster for dorsal and palmer
        kmeans = KMeans(k=k)
        dorsal_centroids, dorsal_clusters = kmeans.fit(dorsal_data)
        palmar_centroids, palmar_clusters = kmeans.fit(palmar_data)

        # K Nearest Neighbours on the Cluster Centroids
        dorsal_centroids = [(x, 'dorsal') for x in dorsal_centroids]
        palmar_centroids = [(x, 'palmar') for x in palmar_centroids]
        centroids = dorsal_centroids + palmar_centroids

        # Fetch the test data set and transform them into feature space
        test_ids, test_data = self.fetchTestData()
        test_data = np.asarray(test_data)

        # Fetch the labels for the images
        cur = self.conn.cursor()
        test_ids_st = str(set(test_ids)).replace('{', '(')
        test_ids_st = test_ids_st.replace('}', ')')
        cur.execute("SELECT image_id, aspect FROM img_meta WHERE image_id IN {0}".format(test_ids_st))
        labels = cur.fetchall()
        cnt_d = 0
        cnt_p = 0
        cnt = 0
        for idx, test in enumerate(test_data):
            cnt_d = 0
            cnt_p = 0
            distances = sorted([(self.l2Dist(test, x[0]), x[1]) for x in centroids], key=lambda x:x[0])[0:9]
            for _, label in distances:
                if label == 'dorsal':
                    cnt_d = cnt_d + 1
                else:
                    cnt_p = cnt_p + 1
            if cnt_p > cnt_d:
                pred_label = 'palmar'
            else:
                pred_label = 'dorsal'
            print(pred_label, labels[idx][1])
            if pred_label == labels[idx][1]:
                cnt = cnt + 1

        print('Accuracy: ', cnt/len(test_data))

c = classify()
c.clusterClassify()
