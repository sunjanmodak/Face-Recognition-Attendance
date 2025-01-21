import cv2
import os
import time
from credentials import datasetpath


# setting dataset directory in present working directory
if not os.path.isdir(datasetpath):
    os.mkdir(datasetpath)

def captureImages(id, studentName, imageCount):

    face_detector = cv2.FaceDetectorYN_create(
                        'cascade/face_detection_yunet_2023mar.onnx',
                        "", 
                        (640, 480),
                        score_threshold=0.5
                        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Unable to access the camera.")
        return

    start_time = time.time()
    count = 0

    print("Starting image capture. Press 'q' to stop.")

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= count + 1:  # Capture every 1 second
            
            _, faces = face_detector.detect(frame)
            if faces is not None:    
                for face in faces:
                    x = face[0].astype('int32')
                    y = face[1].astype('int32')
                    w = face[2].astype('int32')
                    h = face[3].astype('int32')
                    
                    # Save the captured image into the datasets folder
                    face = gray[y:y + h, x:x + w]
      
                    try:
                        count += 1
                        face = cv2.resize(face, (200, 200), interpolation=cv2.INTER_AREA) # Resize to a standard size
                        cv2.imwrite(f"{datasetpath}/{studentName}." + str(id) + '.' + str(count) + ".jpg", face)
                        print(f" Captured {studentName}.{id}.{count}.jpg")
                                      
                    except cv2.error:
                        continue
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
                    
        cv2.imshow("Capturing Images", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time >= imageCount:
            break

    cv2.destroyAllWindows()

def clearDatasetFiles(path):
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(" [INFO] All files in dataset directory deleted successfully.")
    except OSError:
        print(" [INFO] Error occurred while deleting files.")



# main 
clearDatasetInput = input(' [INPUT] Do you want to clear the dataset folder (y/n) (default n) :')
if clearDatasetInput.lower() == 'y':
    clearDatasetFiles(datasetpath)
else:
    print(' [INFO] Skipping dataset files deletion.')

while True:
    print(' Choose option for dataset\n (0) Create dataset  (1) Exit')
    option = int(input('\n [INPUT] Choose Option: '))
    if option == 0:
        try:
            face_id = input('\n [INPUT] Enter student id : ')
            student_name = input(' [INPUT] Enter student name : ')
            imageCount = int(input(' [INPUT] Enter the number of samples to take : '))
            captureImages(face_id, student_name, imageCount)
        except KeyboardInterrupt:
            print('\n [INFO] Exiting...')
            exit()

    elif option == 1:
        print(' [INFO] Exiting...')
        break
    else:
        print(' [INFO] Wrong option please try again...')

