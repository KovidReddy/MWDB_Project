"""
Authors: Kovid, Tharun, Vishal, Anh, Dhriti, Rinku
Last Edited By: Kovid
Last Edited On: 9/22/2019
Class Description: Class to Extract Features from images
"""
# Import statements
import matplotlib.image as mpimg
import glob
import numpy as np
from scipy.stats import skew
from PostgresDB import PostgresDB
import tqdm
import cv2
from skimage import feature
from skimage.transform import downscale_local_mean
import joblib
import json
# Task 3 4 5
import csv
import matplotlib.pyplot as plt
import os

class imageProcess:
    def paths(self):
        try:
            with open('paths.json') as f:
                js = json.load(f)
                dataset = js['dataset']
                model = js['model']
                meta = js['meta']
                ogpath = js['ogpath']
                outputs = js['outputs']
                test = js['test']

        except Exception as error:
            print(error)
            exit(-1)

        return dataset, model, meta, ogpath, outputs, test

    def __init__(self, ext='*.jpg'):
        self.dirpath, self.modelpath, self.metapath, self.ogpath, self.outpath, self.testpath = self.paths()
        self.ext = ext

    # Method to fetch images as pixels
    def fetchImagesAsPix(self, filename):
        image = cv2.imread(filename)
        size = image.shape
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        return img_yuv, size

    # Method to calculate the moments
    def calMommets(self, calc):
        calc = np.array([x for y in calc for x in y])
        mean = np.mean(calc, axis=0)
        sd = np.std(calc, axis=0)
        skw = skew(calc, axis=0)
        mom = [mean.tolist(), sd.tolist(), skw.tolist()]
        mom = [x for y in mom for x in y]
        return mom

    # Method to split image into 100*100 blocks
    def imageMoments(self, image, size, x=100, y=100):
        momments = []
        for idx1 in range(0, size[0], x):
            for idx2 in range(0, size[1], y):
                window = image[idx1:idx1 + x, idx2:idx2 + y]
                momments.append(self.calMommets(window.tolist()))
        return momments

    # Function to calculate the N SIFT feature vectors for each image
    def sift_features(self, filepath):
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return des

    # Function to Calculate the HOG of an image
    def hog_process(self, filename):
        image = cv2.imread(filename)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dsimg = downscale_local_mean(img, (10, 10))
        (H, hogImage) = feature.hog(dsimg, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), block_norm="L2-Hys",
                                    visualize=True)
        return H

    # Function to calculate the local binary pattern of the window
    def calculate_lbp(self, curr_window):
        eps = 1e-7
        hist = []
        # Initializing LBP settings - radius and number of points
        radius = 3
        num_of_points = 8 * radius
        # Checking for uniform patterns
        window_lbp = feature.local_binary_pattern(curr_window, num_of_points, radius, method='uniform')
        # Generating the histogram
        (histogram, temp) = np.histogram(window_lbp.ravel(),
                                         bins=np.arange(0, num_of_points + 3),
                                         range=(0, num_of_points + 2))
        # Typecasting histogram type to float
        histogram = histogram.astype('float')
        # Normalizing the histogram such that sum = 1
        histogram /= (histogram.sum() + eps)
        hist.append(histogram)
        return hist

    # Function to pre-process images into grayscale and form windows of 100X100 to be fed to calculate_lbp
    def lbp_preprocess(self, filename):
        local_binary_pattern = []
        # Converting the BGR image to Grayscale
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)
        for i in range(0, gray.shape[0], 100):
            j = 0
            while j < gray.shape[1]:
                current_window = gray[i:i + 99, j:j + 99]
                temp_lbp = self.calculate_lbp(current_window)
                local_binary_pattern.extend(temp_lbp)
                j = j + 100

        local_binary_pattern = [x for y in local_binary_pattern for x in y]
        local_binary_pattern = np.asarray(local_binary_pattern, dtype=float).tolist()

        return local_binary_pattern

    """
    Method to Save feature data to Postgres Database
    1. Sift: imagedata_s(imageid, data)
    2. Moments: imagedata_m(imageid, data)
    3. Hog: imagedata_h(imageid, data)
    4. LBP: imagedata_l(imageid, data)    
    """
    def dbSave(self, conn, model):
        # Count the number of files in the directory
        filecnt = len(glob.glob(self.dirpath + self.ext))
        pbar = tqdm.tqdm(total=filecnt)
        # Read images from the directory
        for filename in glob.glob(self.dirpath + self.ext):
            if model == 'm':
                pixels, size = self.fetchImagesAsPix(filename)
                momments = self.imageMoments(pixels, size)
                # Convert to string to insert into DB as an array
                values_st = str(np.asarray(momments).tolist())
                dbname = 'imagedata_m'
            elif model == 's':
                des = self.sift_features(filename)
                values_st = str(np.asarray(des).tolist())
                dbname = 'imagedata_s'
            elif model == 'h':
                h_val = self.hog_process(filename)
                values_st = str(np.asarray(h_val).tolist())
                dbname = 'imagedata_h'
            elif model == 'l':
                lbp_val = self.lbp_preprocess(filename)
                values_st = str(np.asarray(lbp_val).tolist())
                dbname = 'imagedata_l'
            else:
                print('Incorrect value for Model provided')
                exit()
            sql = "CREATE TABLE IF NOT EXISTS {db} (imageid TEXT NOT NULL, imagedata TEXT, PRIMARY KEY (imageid))".format(db=dbname)
            cur = conn.cursor()
            cur.execute(sql)
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            # create a cursor
            sql = "SELECT {field} FROM {db} WHERE {field} = '{condition}';".format(field="imageid",db=dbname,condition=name)
            cur.execute(sql)

            if cur.fetchone() is None:
                sql = "INSERT INTO {db} VALUES('{x}', '{y}');".format(x=name,y=values_st, db=dbname)
            else:
                sql = "UPDATE {db} SET {x} ='{y}' WHERE IMAGEID = '{z}'".format(x='imagedata',y=values_st, z= name, db=dbname)
            
            cur.execute(sql)
            conn.commit()
            # close cursor
            cur.close
            pbar.update(1)

    # Method to fetch data from Database
    def dbFetch(self, conn, dbname, condition = ""):
        # Create cursor
        cur = conn.cursor()
        sql = "SELECT * FROM {db} {condition}".format(db=dbname, condition = condition)
        # print (sql)
        cur.execute(sql)
        recs = cur.fetchall()
        return recs

    # Method to access the database
    def dbProcess(self, process='s', model='s', dbase = 'imagedata_h'):
        # Connect to the database
        db = PostgresDB()
        conn = db.connect()
        if process == 's':
            self.dbSave(conn, model)
            print('Data saved successfully to the Database!')
        elif process == 'f':
            recs = self.dbFetch(conn,dbase)
            recs_flt = []
            # Flatten the data structure and 
            for rec in recs:
                recs_flt.append((rec[0],np.asarray(eval(rec[1]))))
            return recs_flt

    # Method to calculate the Cosine Similarity
    def cosine_sim(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return ((1 + (dot_product / (norm_a * norm_b)))/2)*100

    # method to calculate Manhattan distance
    def man_dist(self, vec1, vec2):
        dist = [abs(x - y) for x,y in zip(vec1, vec2)]
        return sum(dist)

    # Calculate the L2 distance
    def l2Dist(self, d1, d2):
        d1 = np.array(d1, dtype=np.float32)
        d2 = np.array(d2, dtype=np.float32)
        dist = cv2.norm(d1, d2, cv2.NORM_L2)
        return dist

    def cosine_similarity(self, imageA, imageB):
        return np.dot(imageA, imageB)/(np.sqrt(np.sum(imageA ** 2, axis=0))*np.sqrt(np.sum(imageB ** 2, axis=0)))
    
    # Calculate the Euclidean distance
    def euclidean_distance(self, imageA, imageB):
        return np.sqrt(np.sum((imageA - imageB) ** 2, axis=0))

    # Calculate the vector matches
    def knnMatch(self, d1, d2, k=2):
        distances = []
        for d in d1:
            dis = sorted([self.l2Dist(d, x) for x in d2])
            distances.append(dis[0:k])
        return distances

    # Method to calculate Similarity for SIFT vectors
    def sift_sim(self, d1, d2):
        matches = self.knnMatch(d1, d2, k=2)
        good = []
        all = []
        d1 = np.array(d1, dtype=np.float32)
        for m, n in matches:
            all.append(m)
            if m < 0.8 * n:
                good.append(m)
        return len(good) / d1.shape[0]

    # Method to calculate Similarity
    def SimCalc(self, img, recs, imgmodel='m', k=5):
        # Calculate the Similarity matrix for Moments model
        rec_dict = dict((x, y) for x, y in recs)
        img_vec = rec_dict[img]
        if imgmodel == 'm':
            sim_matrix = sorted([(rec[0], self.cosine_sim(img_vec, rec[1])) for rec in recs
                                if rec[0] != img], key=lambda x: x[1])
        if imgmodel == 's':
            sim_matrix = sorted([(rec[0], self.sift_sim(img_vec, rec[1])) for rec in recs
                                if rec[0] != img], key=lambda x: x[1], reverse=True)
        return sim_matrix[0:k]


    def queryImageNotLabel(self, image_data, feature, technique, label):
        print("Not Same Label")
        image_data = np.asarray(eval(image_data[0][1]))
        path = self.modelpath

        model = joblib.load(path + os.sep + "{0}_{1}_{2}.joblib".format(feature, technique, label))
        
        if feature == 's' or (feature == 'm' and technique in ("nmf", "lda")):
            latent = np.asarray(model.components_)
            kmeans = joblib.load(path + os.sep + 'kmeans_{0}_{1}_{2}.joblib'.format(latent.shape[1],feature,label))
            histo = np.zeros(latent.shape[1])
            nkp = np.size(image_data)
            for d in image_data:
                idx = kmeans.predict([d])
                histo[idx] += 1/nkp
            image_data = np.asarray(histo)
        image_data = model.transform([image_data])[0]
        return image_data
        
    def similarity(self, feature, technique, dbase, k, image, label = ""):
        db = PostgresDB()
        conn = db.connect()
        if conn is None:
            print("Can not connect to database")
            exit()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM " + dbase)
        data = cursor.fetchall()
        image_id = [rec[0] for rec in data]
        similarity = {}
        if image in image_id:
            image_index = image_id.index(image)
            image_data = np.asarray(eval(data[image_index][1]))
        else:
            print("Not Same Label")
            dbase = 'imagedata_' + feature
            label = label.replace(" ", "_")
            image_data = self.dbFetch(conn,dbase, "WHERE imageid = '{0}'".format(image))
            image_data = self.queryImageNotLabel(image_data, feature, technique, label)
            similarity[image] = self.euclidean_distance(image_data,image_data)
            
        # print (image_id)
        for i in range(len(image_id)):
            image_cmp = np.asarray(eval(data[i][1]))
            similarity[image_id[i]] = self.euclidean_distance(image_data,image_cmp)
        similarity = sorted(similarity.items(), key = lambda x : x[1], reverse=False)
        self.dispImages(similarity,feature, technique, 11, k, label)

    # Method to display images
    def dispImages(self, similarity, feature, technique, no_images, k, label):
        columns = 4
        rows = no_images // columns
        if no_images  % columns != 0:
                rows += 1
        ax = []
        fig=plt.figure(figsize=(30, 20))
        fig.canvas.set_window_title('Task 3 - Images Similarity - Euclidean')
        fig.suptitle(str(no_images - 1) + ' Similar Images of ' + similarity[0][0] + ' based on ' + feature + ", "+ str(k) + " latent semantics and " + technique + " " + label)
        plt.axis('off')
        f=open("../Phase1/Outputs/task_result.txt","w+")
        f.write("Task - Matching Score " + str(no_images) + " images with " + similarity[0][0] + ' based on ' + feature + ", "+ str(k) + " latent semantics and " + technique + ":\n")
        for i in range(no_images):
            f.write(similarity[i][0] + ": " + str(similarity[i][1]) + "\n")
            img = mpimg.imread(self.dirpath + self.ext.replace('*', similarity[i][0]))
            # create subplot and append to ax
            ax.append( fig.add_subplot(rows, columns, i+1))
            if i == 0:
                    ax[-1].set_title("Given Image: " +similarity[i][0])  # set title
            else:
                    ax[-1].set_title("Image "+str(i) + ": " +similarity[i][0] + '\nScore: ' + str(float(similarity[i][1])))  # set title
            ax[-1].axis('off')
            plt.imshow(img)
        plt.savefig('../Phase1/Outputs/task_result.png')
        f.close()
        plt.show()
        plt.close()

    # Method to write to a file
    def writeFile(self, content, path):
        with open(path, 'w') as file:
            file.write(str(content))

    # Convert list to string
    def list2string(self, lst):
        values_st = str(lst).replace('[[', '(')
        values_st = values_st.replace('[', '(')
        values_st = values_st.replace(']]', ']')
        values_st = values_st.replace(']', ')')
        return values_st
    
    def createInsertMeta(self, conn):
        # Read the metadata file
        metafile = self.readMetaData()
        # Create cursor
        cur = conn.cursor()
        # Create the meta table
        cur.execute('DROP TABLE IF EXISTS img_meta')
        sqlc = "CREATE TABLE IF NOT EXISTS " \
               "img_meta(subjectid TEXT, image_id TEXT, gender TEXT, aspect TEXT, orient TEXT, accessories TEXT)"
        cur.execute(sqlc)
        conn.commit()
        # Insert the meta data into the database table
        values_st = self.list2string(metafile)
        sqli = "INSERT INTO img_meta VALUES {x}".format(x=values_st)
        cur.execute(sqli)
        conn.commit()
        print('Meta Data saved into Database!')
        cur.close()
    
    
    def readMetaData(self):
        with open(self.metapath, 'r') as file:
            csv_reader = csv.reader(file)
            meta_file = []
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                sub_id = row[0]
                id = row[7].split('.')[0]
                gender = row[2]
                orientation = row[6].split(' ')
                accessories = row[4]
                meta_file.append([sub_id, id, gender, orientation[0], orientation[1], accessories])
            return meta_file

    def CSV(self, conn, dbase, label = ""):
        label = label.lower()

        if label in ['left', 'right']:
            field = 'orient'
        elif label in ['dorsal', 'palmar']:
            field = 'aspect'
        elif label in ['with accessories', 'without accessories']:
            field = 'accessories'
            if label == 'with accessories':
                label = '1'
            else:
                label = '0'
        elif label in ['male', 'female']:
            field = 'gender'

        cur = conn.cursor()
        sqli = "SELECT image_id, imagedata from img_meta INNER JOIN {db} ON image_id = imageid WHERE {field} = '{label}'".format(field=field, label=label, db = dbase)
        cur.execute(sqli)
        filteredImage = [(x[0],np.asarray(eval(x[1]))) for x in cur.fetchall()]
        return filteredImage



