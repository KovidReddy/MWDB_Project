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


Notice:
We should create the Models folder before running tasks. It should be same level as MWDB_Project folder
HandInfo.csv file should be put inside the working path
