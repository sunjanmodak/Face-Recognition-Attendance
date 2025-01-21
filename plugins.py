import mysql.connector
import cv2 

# connecting to the database
def connect(hostname, username, password, databasename):
    # connecting with the database at hostname as username
    try:
        conn = mysql.connector.connect(
            host = hostname,
            user = username,
            passwd = password,
            database = databasename)

        # checking cunnection status
        if conn.is_connected():
            print(' [INFO] Successfully connected to the database')
            return conn
        else:
            print(f' [INFO] Error connecting to the mysql database at {hostname}')
            return None
    except mysql.connector.errors.InterfaceError:
        print(' [INFO] Network error or host does not exist. Exiting...')
        exit()


def process(grayImage):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    resized = cv2.resize(grayImage, (200, 200), interpolation=cv2.INTER_AREA)         
    normalised = cv2.normalize(resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    final = clahe.apply(normalised)
    return final
