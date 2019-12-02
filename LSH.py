import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imageProcess import imageProcess
import json
import tqdm
class LSH(imageProcess):
    def __init__(self, L=3, k=5):
        self.L = L
        self.k = k
        super().__init__(ext='*.jpg')

    # Generate Random vectors
    def get_random_vectors(self, dim):
        random_vectors = {}
        for i in range(self.L):
            for j in range(self.k):
                random_vectors[i, j] = np.random.randint(0, 9, size=(2, dim)) * 0.1
        return random_vectors

    # Generate projection distances on the lines
    def gen_projections(self, data, label, vectors):
        project_distances = {}
        for idx in vectors:
            projection = []
            for i, img in enumerate(data):
                # A and B are endpoints of the line
                a = np.array(vectors[idx], dtype=np.float)
                a, b = a[0], a[1]
                p = np.array(img, dtype=np.float)
                ap = p - a
                ab = a - b
                # Project points onto the line
                projection.append(np.array(np.around((a + np.dot(ab, ap) / np.dot(ab, ab) * ab), decimals=3)))

            # Get the minimum point so that distances can be calculated from there
            min_proj = np.amin(projection, axis=0)
            dist_dict = {}
            for j, pj in enumerate(projection):
                dist_dict[label[j]] = cv2.norm(min_proj, pj, cv2.NORM_L2)
            project_distances['h' + str(idx)] = dist_dict

        return project_distances

    # Generate Hash Tables
    def gen_hash_tables(self, projections, num_of_buckets=5):
        buckets = {}
        pbar = tqdm.tqdm(total=len(projections))
        for proj in projections:
            projection = projections[proj]
            distances = np.around(np.array(list(projection.values()), dtype=np.float), decimals=3)
            dist = (np.amax(distances) - np.amin(distances)) / num_of_buckets
            bucket_range = np.arange(np.amin(distances), np.amax(distances), dist)
            bucket = {}
            for image, dis in projection.items():
                j = -1
                for b in bucket_range:
                    if np.around(dis, decimals=3) < b:
                        if str(j) not in bucket and j != -1:
                            bucket[str(j)] = []
                        bucket[str(j)].append(image)
                        break
                    j = j + 1
            buckets[proj] = bucket
            pbar.update(1)
        return buckets

    # Save index structure to JSON file
    def saveIndex(self, index_structure):
        try:
            with open(self.outpath + 'index.json', 'w') as j:
                json.dump(index_structure, j)
            print('Saved Index structure to {0} successfully!'.format(self.outpath))
        except Exception as e:
            print(e)

    def fetchIndex(self):
        try:
            with open(self.outpath + 'index.json', 'r') as j:
                index = json.load(j)
            return index
        except Exception as e:
            print('Prcessing Index Structure!')
            return -1


    # Fit LSH method
    def fit(self, data, labels):

        index = self.fetchIndex()
        if index == -1:
            # Generate Random vectors
            random_vectors = self.get_random_vectors(len(data[0]))
            print('Generated Random Vectors!')
            # Generate Projections
            projections = self.gen_projections(data, labels, random_vectors)
            print('Calculated Projections!')
            # Generate Hash Tables
            index_structure = self.gen_hash_tables(projections)
            print('Generated Hash Tables!')
            # Save Index structure to file
            self.saveIndex(index_structure)

        else:
            index_structure = index

        return index_structure

    def display_images(self, images):
        no_images = len(images)
        columns = 4
        rows = no_images // columns
        if no_images % columns != 0:
            rows += 1
        fig = plt.figure(figsize=(30, 20))
        ax = []
        fig.canvas.set_window_title('LSH NN Search')
        for i in range(no_images):
            img = mpimg.imread(self.ogpath + self.ext.replace('*', images[i]))
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i+1))
            if i == 0:
                    ax[-1].set_title("Given Image: " + images[i])  # set title
            else:
                    ax[-1].set_title("Image "+str(i) + ": " + images[i])  # set title
            ax[-1].axis('off')
            plt.imshow(img)
        plt.savefig(self.outpath + 'LSH_result.png')
        plt.show()

    def NNSearch(self, labels, index_structure, imageid):
        image_set = set()
        num_total_images = 0
        temp_set = set(labels)
        for i in range(self.L):
            for j in range(self.k):
                hsh = 'h(' + str(i) + ', ' + str(j) + ')'
                for bucket in index_structure[hsh]:
                    if imageid in index_structure[hsh][str(bucket)]:
                        temp_set = temp_set.intersection(set(index_structure[hsh][str(bucket)]))
                    num_total_images = num_total_images + len(temp_set)
                image_set = image_set.union(temp_set)
        return list(image_set)







# l = LSH()
# x= l.get_random_vectors(256)
# print(len)
