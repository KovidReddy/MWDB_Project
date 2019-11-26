from PostgresDB import PostgresDB
from dimReduction import dimReduction
from dimReduction import KMeans_SIFT
from clustering import KMeans
import glob
import numpy as np
import os
import SVM as sv
from SVM import SVM
from DecisionTree import DecisionTree
import joblib

class classify(dimReduction):
    def __init__(self, feature='l', dim='pca', k=20):
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
            if self.feature == 'm':
                data.append(np.asarray(eval(img[1])).reshape((-1)))
            else:
                data.append(np.asarray(eval(img[1])))
            ids.append(img[0])
            # data.append(eval(img[1]))
        return ids, np.asarray(data)

    # Fetch Test data and insert into database
    def fetchTestData(self):
        db = 'imagedata_test_' + self.feature
        cur = self.conn.cursor()
        # Check if table already exists
        cur.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = '%s')" % (db))
        tflg = cur.fetchone()
        if not tflg[0]:
            values = []
            names = []
            # Fetch images as pixels
            for filename in glob.glob(self.testpath + self.ext):
                if self.feature == 'm':
                    pixels, size = self.fetchImagesAsPix(filename)
                    val = self.imageMoments(pixels, size)
                    # Convert to string to insert into DB as an array
                elif self.feature == 's':
                    val = self.sift_features(filename)
                elif self.feature == 'h':
                    val = self.hog_process(filename)
        
                elif self.feature == 'l':
                    val = self.lbp_preprocess(filename)
                
                else:
                    print('Incorrect value for Model provided')
                    exit()
            
                values.append(str(np.asarray(val).tolist()))
                name = os.path.basename(filename)
                name = os.path.splitext(name)[0]
                names.append(name)
            values_zip = str(set(zip(names, values))).replace('{', ' ')
            values_zip = values_zip.replace('}', ' ')
     
            cur.execute("CREATE TABLE {0} "
                        "AS WITH t(imageid, imagedata) AS (VALUES {1}) SELECT * FROM t".format(db, values_zip))

        # Read data from Table
        cur.execute("SELECT imageid, imagedata, aspect FROM {db}, {meta} WHERE {db}.imageid = {meta}.image_id".format(db="imagedata_test_" + self.feature , meta ="img_meta"))
        data = cur.fetchall()
        test_data = []
        test_ids = []
        test_label = []
        for rec in data:
            if self.feature == 'm':
                test_data.append(np.asarray(eval(rec[1])).reshape((-1)))
            else:
                test_data.append(np.asarray(eval(rec[1])))
            # data_test.append(np.asarray(eval(rec[1])).reshape((-1)))
            test_ids.append(rec[0])
            test_label.append(rec[2])

        # imgs_test = np.append(np.asarray(test_data), np.asarray(test_label).reshape((-1,1)), axis=1)
        # imgs_test = np.append(np.asarray(test_ids).reshape((-1,1)), np.asarray(imgs_test), axis=1)

        self.conn.commit()
        cur.close()
        return np.asarray(test_data), test_ids, test_label

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

    # Function to use SVM to classify images as palmar and dorsal
    def svmClassify(self):
        # Check if tables exist otherwise create them
        self.checkCreate()
        no_clusters = 400
        # Now segregate the content between dorsal and palmar
        dorsal_ids, dorsal_data = self.fetchAspect('dorsal')
        palmar_ids, palmar_data = self.fetchAspect('palmar')

        # svm_data = {}
        # svm_data[-1] = np.asarray(dorsal_data)
        # svm_data[1] = np.asarray(palmar_data)

        svm_data = np.vstack((np.array(dorsal_data), np.array(palmar_data)))
        y1_test = [1.0 for _ in range(len(dorsal_data))]
        y2_test = [-1.0 for _ in range(len(palmar_data))]
        svm_labels = np.hstack((y1_test, y2_test))

        if self.feature == "s":
            imgs_data = []
            print(dorsal_data[0].shape)
            print(np.asarray(palmar_data).shape)
            for data in svm_data:
                imgs_data.extend(data)
            svm_data = np.asarray(imgs_data)
            print(imgs_data[0].shape)

            imgs_data = []
            for data in svm_data:
                imgs_data.extend(data)
            imgs_data = np.asarray(imgs_data)

            Kmeans = KMeans_SIFT(no_clusters)
            clusters = Kmeans.kmeans_process(imgs_data)
            imgs_zip = list(zip(svm_labels, svm_data))
            svm_data = Kmeans.newMatrixSift(imgs_zip, clusters ,'kmeans_s')
            # print(np.asarray(svm_data).shape)
            # imgs_zip = list(zip(imgs_meta, imgs_data))


        # Perform SVM
        svm = SVM(kernel=sv.linear_kernel,C=1000.1)
        svm.fit(svm_data, svm_labels)
        # Fetch the test data set and transform them into feature space
        test_data, test_ids, test_label = self.fetchTestData()

        if self.feature == 's':
            kmeans = joblib.load(self.modelpath + 'kmeans_s.joblib')
            print("aaaaaaaa")
            histo_list = []
            for des in test_data:
                kp = np.asarray(des)
                histo = np.zeros(no_clusters)
                nkp = kp.shape[0]
                for d in kp:
                        idx = kmeans.predict([d])
                        histo[idx] += 1/nkp
                histo_list.append(histo)
            test_data = histo_list
        # Fetch the labels for the images
        # cur = self.conn.cursor()
        # test_ids_st = str(set(test_ids)).replace('{', '(')
        # test_ids_st = test_ids_st.replace('}', ')')
        # cur.execute("SELECT image_id, aspect FROM img_meta WHERE image_id IN {0}".format(test_ids_st))
        # labels = cur.fetchall()
        # print(test_label)
        labels = np.array([-1.0 if x == 'palmar' else 1.0 for x in test_label])

        # Calculate accuracy
        y_pred = svm.predict(test_data)
        correct = np.sum(y_pred == labels)
        print(correct/len(test_data))
        # print(count/len(test_data))
    
    def decisionTreeClassify(self):
        # fetch image dataset
        # label = label.replace(" ", "_")
        # db_feature = 'imagedata_' + feature + '_' + dim
        # _, _ = self.fetchTestData()
        # return
        cur = self.conn.cursor()
        print(self.table_f)
        no_clusters = 400
        
        # sqlm = "SELECT image_id FROM img_meta WHERE subjectid = '{s}'".format(s=subject)
        # image = self.singleImageFetch(img=img, feature=feature)

        # Check for which reduced dimension technique is being used
        # path = self.modelpath
        try:
            model = joblib.load(self.modelpath + self.table_f + '.joblib')
            tree = model[0]
            my_tree = model[1]
        except (OSError, IOError) as e:
            sqlj = "SELECT imageid, imagedata, aspect FROM {db}, {meta} WHERE {db}.imageid = {meta}.image_id".format(db=self.table_f, meta ="img_meta")
            cur.execute(sqlj)
            label_data = cur.fetchall()
            recs_flt = []
            img_meta = []
            label = []
            original = []
            # print(label_data)
            for rec in label_data:
                # print(np.asarray(eval(rec[1])).shape)
                if self.feature == 's':
                    recs_flt.extend(eval(rec[1]))
                    original.append(eval(rec[1]))
                elif self.feature == 'm':
                    recs_flt.append(np.asarray(eval(rec[1])).reshape((-1)))
                else:
                    recs_flt.append(eval(rec[1]))
                img_meta.append(rec[0])
                label.append(rec[2])
            if self.feature == 's':
                
                Kmeans = KMeans_SIFT(no_clusters)
                clusters = Kmeans.kmeans_process(recs_flt)
                imgs_zip = list(zip(img_meta, original))
                recs_flt = Kmeans.newMatrixSift(imgs_zip, clusters ,'kmeans_s')
                print(np.asarray(recs_flt).shape)
                # imgs_zip = list(zip(imgs_meta, imgs_data))
            # return
            # print(np.asarray(recs_flt).shape, len(label))
            imgs_red = np.append(np.asarray(recs_flt), np.asarray(label).reshape((-1,1)), axis=1)
            imgs_red = np.append(np.asarray(img_meta).reshape((-1,1)), np.asarray(imgs_red), axis=1)
            # np.random.shuffle(imgs_red)
            # print(imgs_red)
            # print(imgs_red)
            # print(imgs_red[:150,1:].shape)
            # print(imgs_red[150:].shape)
            # print("Full:", imgs_red[:150])
            # print("Not Full:",imgs_red[:150,1:])
            # exit(1)
            tree = DecisionTree()
            my_tree = tree.build_tree(imgs_red[:,1:])
            with open(self.modelpath + self.table_f + '.joblib', 'wb') as f1:
                joblib.dump([tree, my_tree], f1)

        # sqlj = "SELECT imageid, imagedata, aspect FROM {db}, {meta} WHERE {db}.imageid = {meta}.image_id".format(db="imagedata_test_" + self.feature , meta ="img_meta")
        # cur.execute(sqlj)
        # label_test = cur.fetchall()
        # data_test = []
        # meta_test = []
        # label = []
        data_test, meta_test, label = self.fetchTestData()
        # print(label_data)
        # for rec in label_test:
        #     if self.feature == 'm':
        #         data_test.append(np.asarray(eval(rec[1])).reshape((-1)))
        #     else:
        #         data_test.append(eval(rec[1]))
        #     # data_test.append(np.asarray(eval(rec[1])).reshape((-1)))
        #     meta_test.append(rec[0])
        #     label.append(rec[2])
        if self.feature == 's':
            kmeans = joblib.load(self.modelpath + 'kmeans_s.joblib')
            print("aaaaaaaa")
            histo_list = []
            for des in data_test:
                kp = np.asarray(des)
                histo = np.zeros(no_clusters)
                nkp = kp.shape[0]
                for d in kp:
                        idx = kmeans.predict([d])
                        histo[idx] += 1/nkp
                histo_list.append(histo)
            data_test = histo_list
        print(np.asarray(data_test).shape)
        imgs_test = np.append(np.asarray(data_test), np.asarray(label).reshape((-1,1)), axis=1)
        imgs_test = np.append(np.asarray(meta_test).reshape((-1,1)), np.asarray(imgs_test), axis=1)

        count = 0
        for row in imgs_test:
            # print(row)
            result = tree.print_leaf(tree.classify(row[1:], my_tree))
            d = eval(str(result))
            # print(d)
            if row[-1] in d:
                count += 1
            print ("Image: %s. Actual: %s. Predicted: %s" %
                    (row[0], row[-1], result))
        print("Result:", float(count / len(imgs_test)))

feature = input('Please choose a feature model - SIFT(s), Moments(m), LBP(l), Histogram(h): ')
if feature not in ('s', 'm', 'l', 'h'):
    print('Please enter a valid feature model!')
    exit()
c = classify(feature = feature)

# c.decisionTreeClassify()
c.svmClassify()



# svm = SVM() # Linear Kernel
# svm.fit(data=data_dict)
