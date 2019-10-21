# MWDB_Project
System Specifications:

Python Version: 3.6.5
Postgres DB Version: PostgreSQL 11.5, 64-bit
Operating System in which testing was done: Windows 10

Python Packages needed for running the scripts:

Pillow			v6.1.0	
configparser		v3.8.1
cycler			v0.10.0	
glob2			v0.7	
kiwisolver		v1.1.0	
matplotlib		v3.1.1	
numpy			v1.17.2	
opencv-contrib-python	v3.4.2.16	
opencv-python		v3.4.2.16	
pip			v19.2.3	
progressbar		v2.5	
psycopg2		v2.8.3
pyparsing		v2.4.2	
python-dateutil		v2.8.0	
scipy			v1.3.1	
setuptools		v39.1.0	
six			v1.12.0	
tqdm			v4.35.0	

Run Instructions:

1. Run the python file runFile.py (Use Pycharm python terminal for best visual results)
2. Enter the Task you would like to run (1,2,3) (1- Save to Database, 2- Fetch Features, 3- Compare Images)
3. Enter the Image ID of interest.
4. Enter the Number of Similar Images you would like to return.
5. Check Output folder for results.

####task1:
    #####Input:
        • path: Full path to jpg images folder. The path should end with '\'.
        • feature model: There are four feature models implemented - SIFT, Moments, LBP, Histogram. Enter one of the s,m,l,h characters to select one.
        • dimensionality reduction technique: There are four dimensionality reduction techniques implemented - PCA, SVD, NMF, LDA. Enter one of the following pca, svd, nmf, lda to select one.
        • k: Enter a number k, which is the number of latent semantics for selected dimensionality reduction technique.
    #####Output:
        • The output of this task is saved in a file in folder Outputs named as Task1_Data_ls_{model}_{dimensionality reduction technique}_{k}.txt. The output is an array of length k of tuples of image id and latent semantic values.
####task2:
    #####Input:
        • path: Full path to jpg images folder. The path should end with '\'.
        • feature model: There are four feature models implemented - SIFT, Moments, LBP, Histogram. Enter one of the s,m,l,h characters to select one.
        • dimensionality reduction technique: There are four dimensionality reduction techniques implemented - PCA, SVD, NMF, LDA. Enter one of the following pca, svd, nmf, lda to select one.
        • image: Enter an image id to find similar images using latent semantics.
        • k: Enter a number k, which is the number of latent semantics for selected dimensionality reduction technique.
    #####Output:
        • This task outputs the L2 norm distances of the image to all other images in the folder in ascending order of distances.
####task3:
    #####Input:
        • path: Full path to jpg images folder. The path should end with '\'.
        • feature model: There are four feature models implemented - SIFT, Moments, LBP, Histogram. Enter one of the s,m,l,h characters to select one.
        • dimensionality reduction technique: There are four dimensionality reduction techniques implemented - PCA, SVD, NMF, LDA. Enter one of the following pca, svd, nmf, lda to select one.
        • label: Select one of the following labels - left-hand, right-hand, dorsal, palmar, with accessories, without accessories, male, female.
        • k: Enter a number k, which is the number of latent semantics to be selected for dimensionality reduction technique.
    #####Output:
        • This task outputs the k latent semantics for the images labelled as label entered by user. This output is stored in file named Task3_Data_ls_{feature}_{technique}_{k}.txt inside folder Outputs.
####task4:
    #####Input:
        • path: Full path to jpg images folder. The path should end with '\'.
        • feature model: There are four feature models implemented - SIFT, Moments, LBP, Histogram. Enter one of the s,m,l,h characters to select one.
        • dimensionality reduction technique: There are four dimensionality reduction techniques implemented - PCA, SVD, NMF, LDA. Enter one of the following pca, svd, nmf, lda to select one.
        • image: Enter an image id.
        • k: Enter a number k, which is the number of latent semantics for selected dimensionality reduction technique.
        • label: Select one of the following labels - left-hand, right-hand, dorsal, palmar, with accessories, without accessories, male, female.
    #####Output:
        • In addition to task 3, this task calculates Matching scores of the given image id as input to other images in the folder and find m similar images. L2 norm distance metric is used to find the similarity.
####task5:
    #####Input:
        • path: Full path to jpg images folder. The path should end with '\'.
        • feature model: There are four feature models implemented - SIFT, Moments, LBP, Histogram. Enter one of the s,m,l,h characters to select one.
        • image: Enter an image id of unlabelled image to find label.
        • dimensionality reduction technique: There are four dimensionality reduction techniques implemented - PCA, SVD, NMF, LDA. Enter one of the following pca, svd, nmf, lda to select one.
        • k: Enter a number k, which is the number of latent semantics to be considered for selected dimensionality reduction technique.
        • label: Select one of the following labels - left-hand, right-hand, dorsal, palmar, with accessories, without accessories, male, female.
    #####Output:
        • This task outputs the label of the unlabelled image as one of the following - left-hand vs right-hand, dorsal vs palmar, with accessories vs. without accessories, male vs. female
####task6:
    #####Input:
        • path: Full path to jpg images folder. The path should end with '\'.
        • image: Enter an image id to find most related 3 images.
    #####Output:
        • This task outputs 3 most related images to the image using color moments.
####task7:
    #####Input:
        • path: Full path to jpg images folder. The path should end with '\'.
        • image: Enter an image id
        • k: Enter a number k, which is the number of latent semantics to be considered.
    #####Output:
        • This task outputs similarity matrix for all the images and top k latent semantics for the similarity matrix, sorted in decreasing order.
####task8:
    #####Input:
        • path: Full path to jpg images folder. The path should end with '\'.
        • k: Enter a number k, which is the number of latent semantics to be considered.
    #####Output:
        • This task creates image-metadata matrix and performs NMF on this matrix. Then prints top k latent semantics in decreasing order.

Notice:
We should create the Models and Outputs folder before running tasks. It should be same level as MWDB_Project folder
HandInfo.csv file should be put inside the working path

