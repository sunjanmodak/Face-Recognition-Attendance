# **Automatic Face Recognition Student Attendance System**  

## **Overview**  
The **Automatic Face Recognition Student Attendance System** is a Python-based project that leverages artificial intelligence and computer vision technologies to automate the attendance process. It uses face recognition to identify students in real time and logs their attendance into a MySQL database. This system is designed to replace traditional attendance methods with a more efficient, accurate, and secure solution.  

## **Features**  
- **Real-Time Face Recognition**: Identifies and verifies student faces using live video streams.  
- **Automated Dataset Creation**: Captures and preprocesses facial images of students for training.  
- **Model Training**: Uses the Local Binary Patterns Histograms (LBPH) algorithm for reliable facial recognition.  
- **Database Integration**: Manages student information and attendance records securely in a MySQL database.  
- **Scalability**: Designed to handle large datasets and multiple users.  

---

## **Project Structure**  
```plaintext
├── cascade/  
│   └── face_detection_yunet_2023mar.onnx  # Pre-trained face detection model  
├── credentials.py                        # Database credentials and dataset path  
├── createDataset.py                      # Script for creating the student face dataset  
├── trainModel.py                         # Script for training the face recognition model  
├── detectFace.py                         # Script for real-time face recognition and attendance logging  
├── plugins.py                            # Utility functions for database connection and image preprocessing  
├── trainer/  
│   └── trainer.yml                       # Trained LBPH model file  
├── dataset/                              # Folder containing captured face images  
├── requirements.txt                      # List of Python dependencies  
└──  README.md                             # Documentation for the project

```

---

## **Requirements**  

### **Software**  
- Python 3.8+  
- MySQL Database  
- Libraries:  
  - OpenCV  
  - NumPy  
  - MySQL Connector  

### **Hardware**  
- A computer with a webcam (for real-time face recognition).  

---

## **Installation**  

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/sunjanmodak/Face-Recognition-Attendance.git
   cd Face-Recognition-Attendance
   ```  

2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```  

3. **Set Up Database**:
   - Setup your database 
   - Update `credentials.py` with your MySQL database credentials.  

5. **Run the System**:  
   - Create Dataset:  
     ```bash
     python createDataset.py
     ```  
   - Train Model:  
     ```bash
     python trainModel.py
     ```  
   - Detect Faces and Mark Attendance:  
     ```bash
     python detectFace.py
     ```  

---

## **Usage**  

1. **Dataset Creation**:  
   - Capture face images of students and store them in the `dataset/` folder.  
   - Images are labeled with the format `StudentName.ID.ImageNumber.jpg`.  

2. **Model Training**:  
   - Train the LBPH face recognition model using the captured dataset.  
   - The trained model is saved as `trainer.yml` in the `trainer/` folder.  

3. **Real-Time Recognition**:  
   - Use a webcam to recognize students in real time.  
   - Attendance is logged into the `ATTENDANCE` table in the database.  

---

## **Challenges and Solutions**  

- **Low Accuracy in Dim Lighting**: Enhanced image preprocessing using histogram equalization.  
- **Multiple Faces in the Frame**: Implemented bounding boxes to isolate and detect individual faces.  
- **Scalability Issues**: Optimized the LBPH model for faster processing of larger datasets.  

---

## **Future Enhancements**  

- Integrate a mobile app for real-time attendance notifications.  
- Deploy the system on the cloud for remote access and scalability.  
- Incorporate advanced face recognition models (e.g., CNNs) for higher accuracy.  

---

## **Contributors**  
- **Sunjan Modak** – Developer  
- **Subhankar Mitra (PGT CS)** – Project Guide  

---

## **License**  
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  

---

## **Contact**  
For any queries or suggestions, feel free to reach out at **sunjanmodak01@gmail.com**.
