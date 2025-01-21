import cv2
from datetime import datetime
from plugins import connect, process
from credentials import host, user, passwd, database
import mysql.connector
import numpy as np

# initialising database connection
conn = connect(host, user, passwd, database)
curobj = conn.cursor()

# importing face recogniser data
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

face_detector = cv2.FaceDetectorYN_create(
                        'cascade/face_detection_yunet_2023mar.onnx',
                        "", 
                        (640, 480),
                        score_threshold=0.5
                        )

font = cv2.FONT_HERSHEY_SIMPLEX


# ids: name in dictionary for saved face to be fetched from the database of saved faces
names = {}

# fetching data from the database and loading into names dictionary
curobj.execute('SELECT id_no, sname FROM STUDENTS')
nameData = curobj.fetchall()

print("Known faces...")
for entry in nameData:
    names[entry[0]] = entry[1]
    print(f'{entry[0]} : {entry[1]}')


# recently recognised ids with timestamp

recognised_ids = {}

# fetching ids and timestamp from database for last attendance
curobj.execute('SELECT timestamp, id_no FROM ATTENDANCE')
idData = curobj.fetchall()


for entry in idData:
    recognised_ids[entry[1]] = entry[0].timestamp()
    
print(recognised_ids)


# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height


while True:

    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _, faces = face_detector.detect(img)

    keys = recognised_ids.keys()
    if faces is not None:
        for face in faces:
            x = face[0].astype('int32')
            y = face[1].astype('int32')
            w = face[2].astype('int32')
            h = face[3].astype('int32')
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            try:
                final = process(gray[y:y+h,x:x+w])
                id_no, confidence = recognizer.predict(final)
            except cv2.error:
                continue

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                id = names[id_no]
                confidence_show = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence_show = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence_show), (x+5,y+h-5), font, 1, (255,255,0), 1)

            if ((id_no not in keys) or ((datetime.now().timestamp() - recognised_ids[id_no]) > 300.0)) and round(100 - confidence) > 38:
                try:
                    query = f'INSERT INTO ATTENDANCE VALUES ("{datetime.now()}", {id_no}, {round(100 - confidence)})'
                    curobj.execute(query)
                    print(f"Attendance Marked for {id_no}")
                    conn.commit()
                except mysql.connector.errors.ProgrammingError:
                    print("some Error occured")
            now = datetime.now()
            recognised_ids[id_no] = now.timestamp()

    cv2.imshow('Live Video Feed from camera', img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Cleaning up
print("\n [INFO] Exiting Program and cleanup stuff")
conn.disconnect()
cam.release()
cv2.destroyAllWindows()