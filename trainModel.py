import cv2
import os
import numpy as np
from PIL import Image
from plugins import connect, process
from credentials import host, user, passwd, database, datasetpath
path = datasetpath

if not os.path.isdir('trainer'):
    os.mkdir('trainer')

recognizer = cv2.face.LBPHFaceRecognizer_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# connecting to database


conn = connect(host, user, passwd, database)

curobj = conn.cursor()

# creating STUDENTS table for storing id, name, class
curobj.execute('CREATE TABLE IF NOT EXISTS STUDENTS (id_no INT PRIMARY KEY, name VARCHAR(100) NOT NULL)')

# preparing files for training

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    names = {}

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        img = process(img_numpy)

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        name = os.path.split(imagePath)[-1].split(".")[0]

        if not id in names:
            names[id] = name
            
        faceSamples.append(img)
        ids.append(id)

    return faceSamples, ids, names

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids, names = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

print(' Trained faces...')
for id in names:
    curobj.execute(f'INSERT INTO STUDENTS VALUES ({id}, "{names[id]}")')
    print(f'{id}, {names[id]}')
    conn.commit()

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained with {1} images. Exiting Program".format(len(np.unique(ids)), len(faces)))