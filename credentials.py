import os 


host = os.environ.get('DBHOSTNAME', default = "")

user = os.environ.get('DBUSERNAME', default = "")

passwd = os.environ.get('DBPASSWD', default = "")

database = os.environ.get('DBNAME', default = "")

datasetpath = os.environ.get('DATASETPATH', default = "dataset")
