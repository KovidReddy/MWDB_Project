from PostgresDB import PostgresDB
from dimReduction import dimReduction
from clustering import KMeans
import glob
import numpy as np
import os
import SVM
from SVM import SVM
from LSH import LSH
import tqdm
import math
class classify(dimReduction):
    def __init__(self, feature='l', dim='svd', k=20):
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
    def fetchAspect(self, aspect, red=False):
        cur = self.conn.cursor()
        if red:
            table = self.table_d
        else:
            table = self.table_f
        cur.execute("SELECT imageid, imagedata FROM {0} t1 INNER JOIN img_meta t2 "
                    "ON t1.imageid = t2.image_id AND t2.aspect = '{1}'".format(table, aspect))
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
        # Now segregate the content between dorsal and palmar
        dorsal_ids, dorsal_data = self.fetchAspect('dorsal')
        palmar_ids, palmar_data = self.fetchAspect('palmar')
        svm_data = np.vstack((np.array(dorsal_data), np.array(palmar_data)))
        y1_test = [1.0 for _ in range(len(dorsal_data))]
        y2_test = [-1.0 for _ in range(len(palmar_data))]
        svm_labels = np.hstack((y1_test, y2_test))
        # Perform SVM
        svm = SVM(C=1000.1)
        svm.fit(svm_data, svm_labels)
        # Fetch the test data set and transform them into feature space
        test_ids, test_data = self.fetchTestData()

        # Fetch the labels for the images
        cur = self.conn.cursor()
        test_ids_st = str(set(test_ids)).replace('{', '(')
        test_ids_st = test_ids_st.replace('}', ')')
        cur.execute("SELECT image_id, aspect FROM img_meta WHERE image_id IN {0}".format(test_ids_st))
        labels = cur.fetchall()
        labels = np.array([-1.0 if x == 'palmar' else 1.0 for x in labels])

        # Calculate accuracy
        y_pred = svm.predict(test_data)
        correct = np.sum(y_pred == labels)
        print(correct/len(test_data))

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
        cur.execute("SELECT * FROM imagedata_11K_{0}".format(self.feature))
        data = cur.fetchall()
        img_dict = {}
        for d in data:
            img_dict[d[0]] = eval(d[1])
        self.conn.commit()
        cur.close()
        return img_dict

    # Method to use LSH to classify
    def lshClassify(self, n=10, label='Hand_0000002', k=5, l=3):
        # Check if tables exist otherwise create them
        img_dict = self.fetch11KImages()
        lsh = LSH(L=l, k=k)
        index = lsh.fit(list(img_dict.values()), list(img_dict.keys()))
        neighbors = lsh.NNSearch(list(img_dict.keys()), index, label)

        # Perform Naive KNN
        distances = sorted([(n,self.simMetric(np.array(img_dict[label]), np.array(img_dict[n]))) for n in neighbors], key=lambda x:x[0])[0:n]
        nearest = [x[0] for x in distances]
        print('The Nearest Images to {0} are : '.format(label), nearest)
        lsh.display_images(nearest)

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
        feedback = []
        for x,y in nearest:
            ir = input('{0}: '.format(x))
            if ir == 'R':
                feedback.append(1)
            else:
                feedback.append(-1)
        algo = input('Please choose the algorithm for the feedback mechanism: (SVM -svm, Decision Tree -dt, PPR -ppr, Probability -prob)')

        if algo == 'svm':
            svm = SVM(C=1000.1)
            svm_data = np.array([x[1] for x in nearest])
            svm_labels = np.array(feedback)
            svm.fit(svm_data, svm_labels)
            y_pred = svm.predict([x[1] for x in neighbors])
            indices = [i for i, x in enumerate(y_pred) if x == 1]
            new_neighbors = [neighbors[i] for i in indices]
            distances = sorted([(n, self.simMetric(np.array(label), np.array(n))) for label, n in new_neighbors], key=lambda x: x[0])[0:n]
            new_nearest = [x[0] for x in distances]
            print('The New Nearest Images to {0} are : '.format(label), new_nearest)

        elif algo == 'prob':
            rel_indices = [i for i, x in enumerate(feedback) if x == 1]
            relevant = [nearest[i] for i in rel_indices]
            irel_indices = [i for i, x in enumerate(feedback) if x == 1]
            irrelevant = [nearest[i] for i in irel_indices]
            scores = self.probRelFeedback(relevant, irrelevant, neighbors)
            new_nearest = [x[0] for x in scores[0:n]]
            print('The New Nearest Images to {0} are : '.format(label), new_nearest)

c = classify()
c.relevanceFeedback()
