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
import matplotlib.pyplot as plt
from matplotlib.image import imread
from LSH import LSH
import tqdm
import math
import pandas as pd 
import ppr_helper
import csv

class classify(dimReduction):
    def __init__(self, feature='l', dim='pca', k=30):
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
        no_clusters = 400
        cur = self.conn.cursor()
        cur.execute("SELECT imageid, imagedata FROM {0} t1 INNER JOIN img_meta t2 "
                    "ON t1.imageid = t2.image_id AND t2.aspect = '{1}'".format(self.table_f, aspect))
        imgs = cur.fetchall()
        # Separate Image IDs and Image data
        ids = []
        data = []
        for img in imgs:
            if self.feature == 's' or (self.feature == "m" and self.dim in ("nmf", "lda")):
                data.extend(eval(img[1]))
            elif self.feature == "m":
                data.append(np.asarray(eval(img[1])).reshape((-1)))
            else:
                data.append(np.asarray(eval(img[1])))
            ids.append(img[0])
            # data.append(eval(img[1]))

        if self.feature == "s" or (self.feature == "m" and self.dim in ("nmf", "lda")):
            try:
                # print(self.modelpath + 'kmeans_' + str(no_clusters) + '_' + self.feature + '.joblib')
                kmeans = joblib.load(self.modelpath + 'kmeans_' + str(no_clusters) + '_' + self.feature + '.joblib')
                histo_list = []
                # print("aaaaaaaaaaa")
                for des in imgs:
                    kp = np.asarray(eval(des[1]))
                    histo = np.zeros(no_clusters)
                    nkp = kp.shape[0]
                    for d in kp:
                        idx = kmeans.predict([d])
                        histo[idx] += 1/nkp
                    histo_list.append(histo)
                data = np.asarray(histo_list)
                # print(data.shape)
            except:
                # print("bbbbbbbbbbbbb")

                Kmeans = KMeans_SIFT(no_clusters)
                clusters = Kmeans.kmeans_process(data)
                imgs_zip = [(img[0],np.asarray(eval(img[1]))) for img in imgs]
                data = Kmeans.newMatrixSift(imgs_zip, clusters ,'kmeans_' + str(no_clusters) + "_" + self.feature)
                # print(np.asarray(data).shape)


        return ids, np.asarray(data)

    # Fetch Test data and insert into database
    def fetchTestData(self):
        no_clusters = 400
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
        cur.execute("SELECT imageid, imagedata, aspect FROM {db}, {meta} WHERE {db}.imageid = {meta}.image_id".format(db=db , meta ="img_meta"))
        data = cur.fetchall()
        test_data = []
        test_ids = []
        test_label = []
        for rec in data:
            if self.feature == "s" or (self.feature == "m" and self.dim in ("nmf", "lda")):
                test_data.extend(eval(rec[1]))
            elif self.feature == 'm':
                test_data.append(np.asarray(eval(rec[1])).reshape((-1)))
            else:
                test_data.append(np.asarray(eval(rec[1])))
            test_ids.append(rec[0])
            test_label.append(rec[2])

        # imgs_test = np.append(np.asarray(test_data), np.asarray(test_label).reshape((-1,1)), axis=1)
        # imgs_test = np.append(np.asarray(test_ids).reshape((-1,1)), np.asarray(imgs_test), axis=1)
        
        if self.feature == "s" or (self.feature == "m" and self.dim in ("nmf", "lda")):
            try:
                # print(self.modelpath + 'kmeans_' + str(no_clusters) +  '_' + self.feature + '.joblib')
                kmeans = joblib.load(self.modelpath + 'kmeans_' + str(no_clusters) + '_' + self.feature + '.joblib')
                histo_list = []
                # print("aaaaaaaaaaa")
                for des in data:
                    kp = np.asarray(eval(des[1]))
                    histo = np.zeros(no_clusters)
                    nkp = kp.shape[0]
                    for d in kp:
                        idx = kmeans.predict([d])
                        histo[idx] += 1/nkp
                    histo_list.append(histo)
                test_data = np.asarray(histo_list)
                # print(test_data .shape)
            except:
                # print("bbbbbbbbbbbbb")

                Kmeans = KMeans_SIFT(no_clusters)
                clusters = Kmeans.kmeans_process(test_data)
                imgs_zip = [(img[0],np.asarray(eval(img[1]))) for img in data]
                test_data = Kmeans.newMatrixSift(imgs_zip, clusters ,'kmeans_' + str(no_clusters) + "_" + self.feature)
                print(np.asarray(data).shape)


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

        if self.dim == "svd":
            # Perfrom SVD and get the Feature Latent Semantics for both dorsal and palmar
            _, _, lsDorsal = self.svd(dorsal_data, self.k, self.feature + '_' + self.dim + '_' + 'dorsal')
            _,_, lsPalmar = self.svd(palmar_data, self.k, self.feature + '_' + self.dim + '_' + 'palmar')

        elif self.dim == "pca":
            # Perfrom SVD and get the Feature Latent Semantics for both dorsal and palmar
            _, _, lsDorsal = self.pca(dorsal_data, self.k, self.feature + '_' + self.dim + '_' + 'dorsal')
            _,_, lsPalmar = self.pca(palmar_data, self.k, self.feature + '_' + self.dim + '_' + 'palmar')
        elif self.dim == "nmf":
            _, lsDorsal = self.nmf(dorsal_data, self.k, self.feature + '_' + self.dim + '_' + 'dorsal')
            _, lsPalmar  = self.nmf(palmar_data, self.k, self.feature + '_' + self.dim + '_' + 'palmar')

        elif self.dim == "lda":
            _, lsDorsal = self.lda(dorsal_data, self.k, self.feature + '_' + self.dim + '_' + 'dorsal')
            _, lsPalmar  = self.lda(palmar_data, self.k, self.feature + '_' + self.dim + '_' + 'palmar')
        
        # Fetch the test data set and transform them into feature space
        test_data, test_ids, test_label = self.fetchTestData()
        test_data = np.asarray(test_data)

        # Fetch the labels for the images
        cur = self.conn.cursor()
        test_ids_st = str(set(test_ids)).replace('{', '(')
        test_ids_st = test_ids_st.replace('}', ')')
        cur.execute("SELECT image_id, aspect FROM img_meta WHERE image_id IN {0}".format(test_ids_st))
        labels = cur.fetchall()
        label_dict = {el[0]:el[1] for el in labels}
        cnt = 0
        y_pred = []
        for idx, test in enumerate(test_data):
            dorsalValue = np.linalg.norm(np.dot(test, lsDorsal.T))
            palmarValue = np.linalg.norm(np.dot(test, lsPalmar.T))
            if dorsalValue > palmarValue:
                pred = 'dorsal'
            else:
                pred = 'palmar'
            y_pred.append(pred)
            if label_dict[test_ids[idx]] == pred:
                cnt = cnt + 1
            # print(test_ids[idx])
            # print(label_dict[test_ids[idx]])
            # print('dorsal: ', dorsalValue)
            # print('palmar: ', palmarValue)
        self.visualize(y_pred, test_ids, "task_1_" + self.feature + '_' + self.dim, label = label_dict , accuracy = cnt/len(test_data))

        print('\nAccuracy - {0} - {1}: {2}'.format(self.feature, self.dim,cnt/len(test_data)))

    # Task 2 function to use clustering to classify dorsal palmar
    def clusterClassify(self, k=5):
        # Check if tables exist otherwise create them
        self.checkCreate()
        # Now segregate the content between dorsal and palmar
        dorsal_ids, dorsal_data = self.fetchAspect('dorsal')
        palmar_ids, palmar_data = self.fetchAspect('palmar')

        # Compute Cluster for dorsal and palmer
        kmeans = KMeans(k=k)
        dorsal_centroids, dorsal_clusters = kmeans.fit(dorsal_data)
        palmar_centroids, palmar_clusters = kmeans.fit(palmar_data)
        # print(dorsal_centroids, dorsal_clusters, dorsal_ids)
        # print(palmar_centroids, palmar_clusters, palmar_ids)
        self.visualize(dorsal_clusters, dorsal_ids, "Dorsal " + self.feature.upper(), k=k)
        self.visualize(palmar_clusters, palmar_ids, "Palmar " + self.feature.upper(), k=k)
        # exit(1)
        # K Nearest Neighbours on the Cluster Centroids
        dorsal_centroids = [(x, 'dorsal') for x in dorsal_centroids]
        palmar_centroids = [(x, 'palmar') for x in palmar_centroids]
        centroids = dorsal_centroids + palmar_centroids

        # Fetch the test data set and transform them into feature space
        test_data, test_ids, test_label = self.fetchTestData()
        test_data = np.asarray(test_data)

        # Fetch the labels for the images
        cur = self.conn.cursor()
        test_ids_st = str(set(test_ids)).replace('{', '(')
        test_ids_st = test_ids_st.replace('}', ')')
        cur.execute("SELECT image_id, aspect FROM img_meta WHERE image_id IN {0}".format(test_ids_st))
        labels = cur.fetchall()
        cnt = 0
        y_pred = []
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
            # print(pred_label, labels[idx][1])
            y_pred.append(pred_label)
            if pred_label == labels[idx][1]:
                cnt = cnt + 1
        test_ids = [x[0] for x in labels]
        label_dict = {}
        for x in labels:
            label_dict[x[0]] = x[1]

        self.visualize(y_pred, test_ids, "task_2_" + self.feature, label = label_dict, accuracy = cnt/len(test_data))

        print('Accuracy - {0}: {1}'.format(self.feature,cnt/len(test_data)))

        # print('Accuracy: ', cnt/len(test_data))

    # Function to use SVM to classify images as palmar and dorsal
    def svmClassify(self, constrain = None):
        # print(constrain)
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

        # Perform SVM
        svm = SVM(kernel=sv.linear_kernel)
        svm.fit(svm_data, svm_labels)
        # Fetch the test data set and transform them into feature space
        test_data, test_ids, test_label = self.fetchTestData()
        
        labels = np.array([-1.0 if x == 'palmar' else 1.0 for x in test_label])

        # Calculate accuracy
        y_pred = svm.predict(test_data)
        correct = np.sum(y_pred == labels)
        print(correct/len(test_data))
        y_pred = np.array(['palmar' if x == -1.0 else "dorsal" for x in y_pred])

        check_label_predict = {}
        for i in range(len(test_ids)):
            check_label_predict[test_ids[i]] = test_label[i]

        self.visualize(y_pred, test_ids, "task_4_svm_" + self.feature, label = check_label_predict, accuracy = correct/len(test_data))
        
        # return correct/len(test_data), svm_data, test_data
        return correct/len(test_data), svm_data, test_data
        # print(count/len(test_data))
    
    def decisionTreeClassify(self, depth = 10):
        # fetch image dataset
        # label = label.replace(" ", "_")
        # db_feature = 'imagedata_' + feature + '_' + dim
        # _, _ = self.fetchTestData()
        # return
        cur = self.conn.cursor()
        # print(self.table_f, depth)
        no_clusters = 400
        
        # sqlm = "SELECT image_id FROM img_meta WHERE subjectid = '{s}'".format(s=subject)
        # image = self.singleImageFetch(img=img, feature=feature)

        # Check for which reduced dimension technique is being used
        # path = self.modelpath
        tree = None
        try:
            model = joblib.load(self.modelpath + self.table_f + '_' + str(depth) + '.joblib')
            tree = model[0]
            my_tree = model[1]
        except (OSError, IOError) as e:
            # Now segregate the content between dorsal and palmar
            dorsal_ids, dorsal_data = self.fetchAspect('dorsal')
            palmar_ids, palmar_data = self.fetchAspect('palmar')


            tree_data = np.vstack((np.array(dorsal_data), np.array(palmar_data)))
            y1_test = [1.0 for _ in range(len(dorsal_data))]
            y2_test = [-1.0 for _ in range(len(palmar_data))]
            tree_labels = np.hstack((y1_test, y2_test))
            # print(tree_data.shape)
            # print(tree_labels.shape)
        
            imgs_red = np.append(np.asarray(tree_data), np.asarray(tree_labels).reshape((-1,1)), axis=1)
            
            if not tree:
                tree = DecisionTree(max_depth = depth, min_support = 5)
                my_tree = tree.build_tree(imgs_red)
                with open(self.modelpath + self.table_f + '_' + str(depth) + '.joblib', 'wb') as f1:
                    joblib.dump([tree, my_tree], f1)

        
        data_test, meta_test, test_label = self.fetchTestData()
        
        labels = np.array([-1.0 if x == 'palmar' else 1.0 for x in test_label])

        # print(np.asarray(data_test).shape)
        imgs_test = np.append(np.asarray(data_test), np.asarray(labels).reshape((-1,1)), axis=1)
        imgs_test = np.append(np.asarray(meta_test).reshape((-1,1)), np.asarray(imgs_test), axis=1)

        count = 0
        prediction_label = []
        for row in imgs_test:
            # print(row[-1])
            result = tree.print_leaf(tree.classify(row[1:], my_tree))
            d = eval(str(result))
            # print(d.keys())
            # print(d)
            if float(row[-1]) in d and int(d[float(row[-1])][:-1]) > 50:
                # print("OK")
                count += 1
                # prediction_label.append(row[-1])
            
            if row[-1] == -1.0:
                prediction_label.append("dorsal")
            else:
                prediction_label.append("palmar")
            # print ("Image: %s. Actual: %s. Predicted: %s" %
            #         (row[0], row[-1], result))
        print("Result:", float(count / len(imgs_test)))

        check_label_predict = {}
        for i in range(len(meta_test)):
            check_label_predict[meta_test[i]] = test_label[i]
        self.visualize(prediction_label, imgs_test[:,0], "task_4_tree_" + self.feature, label = check_label_predict, accuracy = float(count / len(imgs_test)))
        return float(count / len(imgs_test))


    def similarity(self, res, id):
            s = {}
            temp = res[id]
            for i in res:
                dist = 1/(1+(math.sqrt(np.sum([((a-b) ** 2) for (a, b) in zip(temp, res[i])]))))
                s[i] = dist
            s = sorted(s.items(), key = lambda x : x[1])
            return s
    
    
    def PPR(self):
        # imgP = imageProcess()
        no_clusters = 400
        res = self.fetchData()
        threshold = 6
        folder = self.testpath
        #meta_data = imgP.readMetaData(meta_file="C:\\Users\\nemad\\Downloads\\MWDB\\project\\phase3_sample_data\\phase3_sample_data\\labelled_set2.csv")
        meta_data_verify = self.readMetaData()
        meta_file= self.metaset2
        with open(meta_file, 'r') as file:
            csv_reader = csv.reader(file)
            meta_file = []
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                sub_id = row[1]
                id = row[8].split('.')[0]
                gender = row[3]
                orientation = row[7].split(' ')
                accessories = row[5]
                meta_file.append([sub_id, id, gender, orientation[0], orientation[1], accessories])

        meta_data = meta_file

        dorsal_ids = []
        palmar_ids = []
        d_personalization = {}
        p_personalization = {}
        p = d = 0
        for m in meta_data:
            if m[3] == 'dorsal':
                dorsal_ids.append(m[1])
                d+=1
            else:
                palmar_ids.append(m[1])
                p+=1
        #print("TRAINING DATA HAS ", p, "PALMAR and ", d, "DORSAL IMAGES")

        for i in res:
                d_personalization[i] = 0

        data = []
        for i in res:
            # print("i:",i)
            weights = ppr_helper.similarity(res, i)
            temp_dict = {}
            for w in weights:
                temp_dict[w[0]] = w[1]
            data.append(temp_dict)
        df = pd.DataFrame(data, index=res.keys())

        for i in res:
            weights = ppr_helper.similarity(res, i)
        img_no = 1
        w = 10
        h = 15
        fig = plt.figure(figsize=(10, 8))
        columns = 10
        rows = 10
        final_Result = {}
        test_data, test_ids, test_label = self.fetchTestData()
        for i in range(len(test_data)):
            temp = df
            # if self.feature == 'm':
            #     pixels, size = self.fetchImagesAsPix(filename)
            #     val = self.imageMoments(pixels, size)
            #         # Convert to string to insert into DB as an array
            # elif self.feature == 's':
            #     val = self.sift_features(filename)
            # elif self.feature == 'h':
            #     val = self.hog_process(filename)
    
            # elif self.feature == 'l':
            #     val = self.lbp_preprocess(filename)
            
            # else:
            #     print('Incorrect value for Model provided')
            #     exit()

            h_val = test_data[i]
            # filename = filename[-16:]
            # filename = filename[0:len(filename)-4]
            filename = test_ids[i]
            # print("file:",filename)
            s = {}
            col = []
            for i in res:
                dist = math.sqrt(np.sum([((a - b) ** 2) for (a, b) in zip(h_val, res[i])]))
                s[i] = dist
                col.append(dist)
            df1 = pd.DataFrame([s], index=[filename])
            temp = pd.concat([temp, df1])
            col.append(0)
            temp[filename] = col
            d_personalization[filename] = 1
            pageRankScores = np.zeros(len(d_personalization))
            teleportation_matrix = np.zeros(len(d_personalization))
            t = 0
            for i in d_personalization:
                pageRankScores[t] = d_personalization[i]
                teleportation_matrix[t] = d_personalization[i]
                t += 1

            pageRankScores = ppr_helper.ppr(temp, pageRankScores, teleportation_matrix, 0.85)
            z = 0
            res1 = {}
            for i in d_personalization:
                if i != filename:
                    res1[i] = pageRankScores[z]
                z += 1
            res1 = sorted(res1.items(), key=lambda x: x[1], reverse=True)
            del d_personalization[filename]
            prob_d = res1[0]
            d = 0
            p = 0
            for tp in range(5):
                cl = res1[tp][0]
                if cl in dorsal_ids:
                    d += res1[tp][1]
                else:
                    p += res1[tp][1]
            if d > p:
                print(filename, "is dorsal with probability", d)
                title = "dorsal"
            else:
                print(filename, "is palmar with probability", p)
                title = "palmar"
            final_Result[filename] = title
            img = imread(folder + filename + ".jpg")
            ax1 = fig.add_subplot(rows, columns, img_no)
            ax1.title.set_text(title)
            plt.imshow(img)
            plt.axis('off')
            img_no += 1
        verification_d = {}
        for i in meta_data_verify:
            verification_d[i[1]] = i[3]

        count = 0
        for i in final_Result:
            if final_Result[i] == verification_d[i]:
                count += 1
        print("accuracy ", count/len(final_Result)*100)
        plt.suptitle("Personalized PageRank Classifier")
        plt.show()

    def fetchData(self, table_name=""):
        no_clusters = 400
        if not table_name:
            table_name = self.table_f
        self.checkCreate()
        cur = self.conn.cursor()
        res = {}
        cur.execute("SELECT imageid, imagedata FROM {0}".format(table_name))
        imgs = cur.fetchall()
        # Separate Image IDs and Image data
        ids = []
        data = []

        for img in imgs:
            if self.feature == 's' or (self.feature == "m" and self.dim in ("nmf", "lda")):
                data.extend(eval(img[1]))
            elif self.feature == "m":
                data.append(np.asarray(eval(img[1])).reshape((-1)))
            else:
                data.append(np.asarray(eval(img[1])))
            ids.append(img[0])
            # data.append(eval(img[1]))

        if self.feature == "s" or (self.feature == "m" and self.dim in ("nmf", "lda")):
            try:
                # print(self.modelpath + 'kmeans_' + str(no_clusters) + '_' + self.feature + '.joblib')
                kmeans = joblib.load(self.modelpath + 'kmeans_' + str(no_clusters) + '_' + self.feature + '.joblib')
                histo_list = []
                # print("aaaaaaaaaaa")
                for des in imgs:
                    kp = np.asarray(eval(des[1]))
                    histo = np.zeros(no_clusters)
                    nkp = kp.shape[0]
                    for d in kp:
                        idx = kmeans.predict([d])
                        histo[idx] += 1/nkp
                    histo_list.append(histo)
                data = np.asarray(histo_list)
                # print(data.shape)
            except:
                # print("bbbbbbbbbbbbb")

                Kmeans = KMeans_SIFT(no_clusters)
                clusters = Kmeans.kmeans_process(data)
                imgs_zip = [(img[0],np.asarray(eval(img[1]))) for img in imgs]
                data = Kmeans.newMatrixSift(imgs_zip, clusters ,'kmeans_' + str(no_clusters) + "_" + self.feature)
                # print(np.asarray(data).shape)


        for i in range(len(ids)):
            # res.append({img[0]: eval(img[1])})
            res[ids[i]] = data[i]
            # ids.append(img[0])
            # data.append(eval(img[1]))
        # print("returning", res,"ending")
        return res
   
   
   
   # Method to use LSH to classify
    def lshClassify(self, n=10, label='Hand_0000002', k=5, l=3):
        # Check if tables exist otherwise create them
        img_dict = self.fetch11KImages()
        lsh = LSH(L=l, k=k)
        index = lsh.fit(list(img_dict.values()), list(img_dict.keys()))
        neighbors = lsh.NNSearch(list(img_dict.keys()), index, label)

        # Perform Naive KNN
        distances = sorted([(n,self.simMetric(np.array(img_dict[label]), np.array(img_dict[n]))) for n in neighbors],reverse=True, key=lambda x:x[1])[0:n]
        nearest = [x[0] for x in distances]
        print('The Nearest Images to {0} are : '.format(label), nearest)
        self.display_images(nearest, 'feedback.png')

        return [(x,img_dict[x]) for x in nearest], [(x,img_dict[x]) for x in neighbors]

    def probRelFeedback(self, relevant, irrelevant, neighbors):
        # First convert the image data into binary values
        bin_data = []
        labels = []
        for d in neighbors:
            threshold = np.mean(d[1])
            bin_data.append([1 if x > threshold else 0 for x in d[1]])
            labels.append(d[0])
        bin_data = list(zip(labels,bin_data))
        relevant_vals = np.array([x[1] for x in relevant])
        irrelevant_vals = np.array([x[1] for x in irrelevant])
        # Now calculate the parameters for probability calculation
        r = relevant_vals.sum(axis=0)
        ir = irrelevant_vals.sum(axis=0)
        R = len(relevant)
        IR = len(irrelevant)
        # Calculate the probability scores for each of the neighbors
        scores = []
        print('Calculating Probability Scores...')
        for lab, n in bin_data:
            prob_score = np.sum([x * math.log((r[i]/(R - r[i])) + 0.5 / ((ir[i]/(IR - ir[i])) + 0.5)) for i, x in enumerate(n)])
            scores.append((lab, prob_score))
        # Sort the images by their probability scores
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores

    # Method to perform Relevance Feedback based ranking
    def relevanceFeedback(self, n=10, label='Hand_0000002', k=5, l=3):
        # Perform Task 5 to get outputs
        nearest, neighbors = self.lshClassify(n=n, label=label, k=k, l=l)

        # Input of each image as Relevant or Irrelevant
        print('Please provide feedback for each of the nearest Images (Relevant - R/ Irrelevant - I): ')
        print('Control Options: Skip image - S / Exit Feedback - E')
        feedback = []
        p_feedback = 0
        p_feedback_imgs = []
        for x,y in nearest:
            ir = input('{0}: '.format(x))
            if ir == 'R':
                feedback.append(1)
                p_feedback_imgs.append(x)
                p_feedback += 1
            elif ir == 'I':
                feedback.append(-1)
            elif ir == 'S':
                feedback.append(0)
            elif ir == 'E':
                break

        algo = input('Please choose the algorithm for the feedback mechanism: '
                     '(SVM -svm, Decision Tree -dt, PPR -ppr, Probability -prob)')

        if algo == 'svm':
            svm = SVM(C=1000.1)
            rel_data = [nearest[i][1] for i,x in enumerate(feedback) if x == 1]
            irel_data = [nearest[i][1] for i,x in enumerate(feedback) if x == -1]
            rel_labels = np.array([1.0 for x in feedback if x == 1])
            irel_labels = np.array([-1.0 for x in feedback if x == -1])
            svm_labels = np.hstack((rel_labels, irel_labels))
            svm_data = np.vstack((np.array(rel_data), np.array(irel_data)))
            svm.fit(svm_data, svm_labels)
            y_pred = svm.predict([x[1] for x in neighbors])
            indices = [i for i, x in enumerate(y_pred) if x == 1]
            new_neighbors = [neighbors[i] for i in indices]
            src_image = [x[1] for x in nearest if x[0] == label]
            distances = sorted([(l, self.simMetric(np.array(src_image[0]), np.array(n))) for l, n in new_neighbors], key=lambda x: x[1], reverse=True)[0:n]
            new_nearest = [x[0] for x in distances]
            print('The New Nearest Images to {0} are : '.format(label), new_nearest)
            self.display_images(new_nearest, 'feedback.png')

        if algo == 'dt':
            
            tree = DecisionTree(max_depth = 10, min_support = 1)
            rel_data = [nearest[i][1] for i,x in enumerate(feedback) if x == 1]
            irel_data = [nearest[i][1] for i,x in enumerate(feedback) if x == -1]
            rel_labels = np.array([1.0 for x in feedback if x == 1])
            irel_labels = np.array([-1.0 for x in feedback if x == -1])
            tree_labels = np.hstack((rel_labels, irel_labels))
            tree_data = np.vstack((np.array(rel_data), np.array(irel_data)))

            # print(tree_labels)
            # tree_data = np.vstack((np.array(dorsal_data), np.array(palmar_data)))
            # y1_test = [1.0 for _ in range(len(dorsal_data))]
            # y2_test = [-1.0 for _ in range(len(palmar_data))]
            # tree_labels = np.hstack((y1_test, y2_test))
            # print(tree_data.shape)
            # print(tree_labels.shape)
        
            imgs_red = np.append(np.asarray(tree_data), np.asarray(tree_labels).reshape((-1,1)), axis=1)

            my_tree = tree.build_tree(imgs_red)

            
        
            # tree = decisionTreeClassify(depth = 5)
            # tree_data = np.array([x[1] for x in nearest])
            # tree_labels = np.array(feedback)

            # imgs_red = np.append(np.asarray(tree_data), np.asarray(tree_labels).reshape((-1,1)), axis=1)

            # my_tree = tree.build_tree(imgs_red)
            
            data_test = [x[1] for x in neighbors]
            test_ids = [x[0] for x in neighbors]

            cur = self.conn.cursor()
            test_ids_st = str(set(test_ids)).replace('{', '(')
            test_ids_st = test_ids_st.replace('}', ')')
            cur.execute("SELECT image_id, aspect FROM img_meta WHERE image_id IN {0}".format(test_ids_st))
            labels = cur.fetchall()
            labels = np.array([-1.0 if x[1] == 'palmar' else 1.0 for x in labels])

            # print(np.asarray(data_test).shape)
            # print(np.asarray(labels).shape)
            imgs_test = np.append(np.asarray(data_test), np.asarray(labels).reshape((-1,1)), axis=1)
            # imgs_test = np.append(np.asarray(test_ids).reshape((-1,1)), np.asarray(imgs_test), axis=1)

            # count = 0
            y_pred = []
            for row in imgs_test:
                # print(row[-1])
                result = tree.print_leaf(tree.classify(row, my_tree))
                d = eval(str(result))
                # print(d.keys())
                # print(d)
                if -1.0 in d and int(d[-1.0][:-1]) > 50:
                    # print("OK")
                    # count += 1
                    y_pred.append(-1)
                else:
                    # print("Have 1")
                    y_pred.append(1)
                # print ("Image: %s. Actual: %s. Predicted: %s" %
                #         (row[0], row[-1], result))
            # print("Result:", float(count / len(imgs_test)))
            # print(y_pred)
            # y_pred = svm.predict([x[1] for x in neighbors])
            indices = [i for i, x in enumerate(y_pred) if x == 1]
            new_neighbors = [neighbors[i] for i in indices]
            src_image = [x[1] for x in nearest if x[0] == label]
            distances = sorted([(l, self.simMetric(np.array(src_image[0]), np.array(n))) for l, n in new_neighbors], key=lambda x: x[1], reverse=True)[0:n]
            new_nearest = [x[0] for x in distances]
            print('The New Nearest Images to {0} are : '.format(label), new_nearest)
            self.display_images(new_nearest, 'feedback.png')

        elif algo == 'prob':
            rel_indices = [i for i, x in enumerate(feedback) if x == 1]
            relevant = [nearest[i] for i in rel_indices]
            irel_indices = [i for i, x in enumerate(feedback) if x == 1]
            irrelevant = [nearest[i] for i in irel_indices]
            scores = self.probRelFeedback(relevant, irrelevant, neighbors)
            new_nearest = [x[0] for x in scores[0:n]]
            print('The New Nearest Images to {0} are : '.format(label), new_nearest)
            self.display_images(new_nearest, 'feedback.png')

        elif algo == 'ppr':
            data = []
            nearest_dict = {}
            personalization = {}
            for i in nearest:
                nearest_dict[i[0]] = i[1]
            for i in nearest_dict:
                weights = ppr_helper.similarity(nearest_dict, i)
                temp_dict = {}
                for w in weights:
                    temp_dict[w[0]] = w[1]
                data.append(temp_dict)
                personalization[i] = 0
            df = pd.DataFrame(data, index=nearest_dict.keys())
            y_pred = []
            for neighbor in neighbors:
                temp = df
                h_val = neighbor[1]
                s = {}
                col = []
                for i in nearest_dict:
                    dist = math.sqrt(np.sum([((a - b) ** 2) for (a, b) in zip(h_val, nearest_dict[i])]))
                    s[i] = dist
                    col.append(dist)
                df1 = pd.DataFrame([s], index=[neighbor[0]])
                temp = pd.concat([temp, df1])
                col.append(0)
                temp[neighbor[0]] = col
                personalization[neighbor[0]] = 1

                pageRankScores = np.zeros(len(personalization))
                teleportation_matrix = np.zeros(len(personalization))
                t = 0
                for i in personalization:
                    pageRankScores[t] = personalization[i]
                    teleportation_matrix[t] = personalization[i]
                    t += 1

                pageRankScores = ppr_helper.ppr(temp, pageRankScores, teleportation_matrix, 0.85)
                print(pageRankScores)
                del personalization[neighbor[0]]

                z = 0
                res1 = {}
                for i in d_personalization:
                    if i != neighbor[0]:
                        res1[i] = pageRankScores[z]
                    z += 1
                res1 = sorted(res1.items(), key=lambda x: x[1], reverse=True)

                prob_d = res1[0]
                d = 0
                p = 0
                for tp in range(5):
                    cl = res1[tp][0]
                    if cl == 1:
                        d += res1[tp][1]
                    else:
                        p += res1[tp][1]
                if d > p:
                    print(filename, "is Relevant with probability", d)
                    title = 1
                else:
                    print(filename, "is Irrelevant with probability", p)
                    title = -1
                y_pred.append(title)
                
            indices = [i for i, x in enumerate(y_pred) if x == 1]
            new_neighbors = [neighbors[i] for i in indices]
            src_image = [x[1] for x in nearest if x[0] == label]
            distances = sorted([(l, self.simMetric(np.array(src_image[0]), np.array(n))) for l, n in new_neighbors], key=lambda x: x[1], reverse=True)[0:n]
            new_nearest = [x[0] for x in distances]
            print('The New Nearest Images to {0} are : '.format(label), new_nearest)
            self.display_images(new_nearest, 'feedback.png')



    def display_images(self, images, title):
        no_images = len(images)
        columns = 4
        rows = no_images // columns
        if no_images % columns != 0:
            rows += 1
        fig = plt.figure(figsize=(30, 20))
        ax = []
        fig.canvas.set_window_title('LSH NN Search')
        for i in range(no_images):
            img = imread(self.ogpath + self.ext.replace('*', images[i]))
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i+1))
            if i == 0:
                    ax[-1].set_title("Given Image: " + images[i])  # set title
            else:
                    ax[-1].set_title("Image "+str(i) + ": " + images[i])  # set title
            ax[-1].axis('off')
            plt.imshow(img)
        plt.savefig(self.outpath + title)
        plt.show()
    
    
    # Fetch 11K images
    def fetch11KImages(self):
        # Check if table exists for 11 K images
        cur = self.conn.cursor()
        # Check if table already exists
        cur.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'imagedata_11k_{0}')".format(self.feature))
        tflg = cur.fetchone()
        if not tflg[0]:
            print('Loading 11K images to the database...')
            # Fetch images as pixels
            filecnt = len(glob.glob(self.ogpath + self.ext))
            pbar = tqdm.tqdm(total=filecnt)
            dbname = 'imagedata_11K_'+self.feature
            for filename in glob.glob(self.ogpath + self.ext):
                if self.feature == 'l':
                    h_val = self.lbp_preprocess(filename)
                elif self.feature == 'h':
                    h_val = self.hog_process(filename)
                values_st = str(np.asarray(h_val).tolist())
                name = os.path.basename(filename)
                name = os.path.splitext(name)[0]
                sql = "CREATE TABLE IF NOT EXISTS {db} (imageid TEXT NOT NULL, imagedata TEXT, PRIMARY KEY (imageid))".format(
                    db=dbname)
                cur.execute(sql)
                # create a cursor
                sql = "SELECT {field} FROM {db} WHERE {field} = '{condition}';".format(field="imageid", db=dbname,
                                                                                       condition=name)
                cur.execute(sql)
                if cur.fetchone() is None:
                    sql = "INSERT INTO {db} VALUES('{x}', '{y}');".format(x=name, y=values_st, db=dbname)
                else:
                    pass
                cur.execute(sql)
                self.conn.commit()
                # close cursor
                pbar.update(1)
        else:
            print('11K imagedata already loaded into the Database')
            # Read data from Table
        print('Fetching the 11k images from the Database...')
        cur.execute("SELECT * FROM imagedata_11K_{0}".format(self.feature))
        data = cur.fetchall()
        img_dict = {}
        for d in data:
            img_dict[d[0]] = eval(d[1])
        self.conn.commit()
        cur.close()
        return img_dict

    def visualize(self, clusters, image_id, title, accuracy = None, k = None, label = None):
        # print(self.dirpath)

        cluster_dic = {}
    
        for i in clusters:
            if i not in cluster_dic:
                cluster_dic[i] = []
        # print(cluster_dic)

        for i in range(len(clusters)):
            cluster_dic[clusters[i]] += [image_id[i]]
        # fig = plt.figure()
        # print(cluster_dic)
        number_of_files = len(image_id)
        openFile = open(self.outpath + title.replace(" ","_").lower() + ".html", "w")
        openFile.write("""<!DOCTYPE html><html><head><style>* {box-sizing: border-box;}.column {
                        float: left;
                        width: 10%;
                        padding: 5px;
                        }

                        /* Clearfix (clear floats) */
                        .row::after {
                        content: "";
                        clear: both;
                        display: table;
                        }
                        </style>
                        </head>
                        <body>""")
        if accuracy:
            openFile.write("<h1>The resulting image of " + title + " with " + str(accuracy) + " accuracy :</h1>")
            # accuracy = 
        else:
            openFile.write("<h1>The resulting image of " + title + ":</h1>")
        for key, value in sorted(cluster_dic.items()):
            # print(key, value)
            number_of_files = len(value)
            # print(number_of_files // 10)
            if key == "palmar" or key == "dorsal":
                openFile.write("<h3>" + key.title() + " Images Prediction </h3>")
            else:
                openFile.write("<h3>Cluster " + str(int(key))+ '</h3>')
            for i in range((number_of_files // 10) + 1):
                openFile.write("""<div class="row">""")
                if (number_of_files - i*10) >= 10:
                    col = 10
                else:
                    col = (number_of_files - i*10)
                for j in range(col):
                    openFile.write("""<div class="column">""")
                    openFile.write(value[i * 10 + j])
                    if k:
                        openFile.write('<img src="' + self.dirpath + value[i * 10 + j] + self.ext[1:] + '" style="width:100%">')
                    else:
                        wrong = '"'
                        if label[value[i * 10 + j]] != key:
                            wrong = ';border-color :red" border = "5"'
                        openFile.write('<img src="' + self.testpath + value[i * 10 + j] + self.ext[1:] + '" style="width:100%' + wrong + '>')
                        
                    openFile.write('</div>')
                    # print(value[i * 10 + j])
                    # with Image.open(self.dirpath + value[i * 10 + j] + self.ext[1:]) as test_image:
                    #     test_image.resize((600,600), Image.ANTIALIAS).show()
                    #     feature = input('Relevant(R) or Irrelevant(I):')
                    #     time.sleep(10)
                    # img = Image.open(self.dirpath + value[i * 10 + j] + self.ext[1:])
                    # img = img.resize((600,600), Image.ANTIALIAS)
                    # img.show(title = value[i * 10 + j])
                    # feature = input('Relevant(R) or Irrelevant(I):')
                    # for proc in psutil.process_iter():
                    #     if proc.name() == "display":
                    #         proc.kill()

                    # a=fig.add_subplot(1,5,i+1)
                    # image = imread(self.dirpath + image_id[i] + self.ext[1:])
                    # a.imshow(image,cmap='Greys_r')
                    # a.axis('off')
                    # a.set_title("abc")
                openFile.write('</div>')
            # plt.show()
        openFile.close()

# feature = input('Please choose a feature model - SIFT(s), Moments(m), LBP(l), Histogram(h): ')
# if feature not in ('s', 'm', 'l', 'h'):
#     print('Please enter a valid feature model!')
#     exit()
# technique = input('Please choose a dimensionality reduction technique - PCA(pca), SVD(svd), NMF(nmf), LDA(lda): ')
# c = classify(feature = feature, dim = technique)
# c.clusterClassify()
# # for f in ["l", "h", "m", "s"]:
# #     # for t in ["svd", "pca", "lda" , "nmf"]:
# #     c = classify(feature = f)
# # # c.relevanceFeedback()
# #     c.clusterClassify()

#         # c.LSAnalysis()
# c = classify(feature = feature)
# # c.PPR()
# exit(1)
# # # c.svmClassify()

# obj_list = []
# x_axis = []
# data = []
# test = []
# for i in range(100,2001,100):
#     # accuracy = c.decisionTreeClassify(i)
#     accuracy, data, test = c.svmClassify(i + 0.1, data, test)
#     # Calculate the Centroids and Clusters
#     # centroids, clusters = kmeans.fit(table, False)
#     # # Create the Cluster matrix for k = 5
#     # if i == 5:
#     #     cluster_mat = [(x,clusters[x]) for x in range(len(table))]
#     # # Calculate the Objective function
#     # ac = kmeans.objective_func(table, centroids, clusters)
#     # Append to objective function list
#     obj_list.append(accuracy)
#     # Create x axis
#     x_axis.append(i + 0.1)

# # Save CSV files
# # with open('clusters.csv', 'w', encoding='utf-8', newline="") as f:
# #     writer = csv.writer(f)
# #     writer.writerows(cluster_mat)
# print(obj_list, x_axis)
# # Plot The objective function vs number of clusters
# plt.plot(x_axis, obj_list)
# plt.xlabel('C')
# plt.ylabel('Accuracy')
# #plt.savefig('graph.png')
# plt.title('SVM - ' + feature + ' - ' + "Labeled Set 2 and Unlabeled Set 2")
# plt.show()

# svm = SVM() # Linear Kernel
# svm.fit(data=data_dict)



