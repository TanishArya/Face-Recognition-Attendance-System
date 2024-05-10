import streamlit as st
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import base64
import uuid

def home():
  st.title("Face Recognition Attendence System")
  
  st.write("A Face Recognition Attendence System is a type of biometric technology that uses AI (Artificial Intelligence) to automatically identify and verify individuals based on their facial characterstics.")
  
  st.write("A face recognition attendance system is a biometric technology that uses artificial intelligence to identify and verify people based on their facial features.")
  
  st.write("It uses a digital camera to capture an image of the face, a computer to process and analyze it, and an output device to display the identification result.")

def register_faces():
    st.title("Face Registration")

    st.write("Please fill in the following details:")
    name = st.text_input("Name")
    enrollment = st.text_input("Enrollment No.")
    branch = st.selectbox("Select Branch", ['CSE', 'ME', 'EX', 'EC', 'CE', 'AIADS'])
    year = st.selectbox("Select Year", ['1', '2', '3', '4'])

    if st.button("Start Face Registration"):
        video = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default (1).xml")

        faces_data = []
        names = []
        enrollments = []
        branches = []
        years = []

        while len(names) < 100:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in detected_faces:
                crop_image = frame[y:y + h, x:x + w, :]
                resized_image = cv2.resize(crop_image, (50, 50))
                faces_data.append(resized_image)
                names.append(name)
                enrollments.append(enrollment)
                branches.append(branch)
                years.append(year)

                cv2.putText(frame, str(len(names)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

            cv2.imshow("Video", frame)
            k = cv2.waitKey(1)
            if k == ord('q') or len(names) == 100:
                break

        video.release()
        cv2.destroyAllWindows()
        st.success("Registration successful.")

        faces_data = np.asarray(faces_data)
        faces_data = faces_data.reshape(len(names), -1)

        if not os.path.exists(f'Attendene Database/{branch}_{year}'):
            os.makedirs(f'Attendene Database/{branch}_{year}')

        if 'names.pkl' not in os.listdir(f'Attendene Database/{branch}_{year}'):
            with open(f'Attendene Database/{branch}_{year}/names.pkl', 'wb') as f:
                pickle.dump(names, f)
        else:
            with open(f'Attendene Database/{branch}_{year}/names.pkl', 'rb') as f:
                existing_names = pickle.load(f)
            names = existing_names + names
            with open(f'Attendene Database/{branch}_{year}/names.pkl', 'wb') as f:
                pickle.dump(names, f)

        if 'faces_data.pkl' not in os.listdir(f'Attendene Database/{branch}_{year}'):
            with open(f'Attendene Database/{branch}_{year}/faces_data.pkl', 'wb') as f:
                pickle.dump(faces_data, f)
        else:
            with open(f'Attendene Database/{branch}_{year}/faces_data.pkl', 'rb') as f:
                existing_faces = pickle.load(f)
            faces_data = np.append(existing_faces, faces_data, axis=0)
            with open(f'Attendene Database/{branch}_{year}/faces_data.pkl', 'wb') as f:
                pickle.dump(faces_data, f)

        pass

def take_attendance(branch, year):
    st.title("Take Attendance")

    st.write("Please look at the camera to take attendance.")
    st.write("For Mark Attendence Press = O")
    st.write("For Exit = Q")
    


    with st.spinner("Initializing camera..."):
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            st.error("Error: Unable to access the camera.")
            return

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default (1).xml")

    # Load names and faces data
    names_file = f'Attendene Database/{branch}_{year}/names.pkl'
    faces_file = f'Attendene Database/{branch}_{year}/faces_data.pkl'

    if not os.path.exists(names_file) or not os.path.exists(faces_file):
        st.error("Error: No Student Records found.")
        return

    with open(names_file, 'rb') as f:
        LABELS = pickle.load(f)

    with open(faces_file, 'rb') as f:
        FACES = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    COL_NAMES = ['Name', 'Date', 'Time']

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            crop_image = frame[y:y + h, x:x + w, :]
            resized_image = cv2.resize(crop_image, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_image)
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            timestamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            exist = os.path.isfile(f'Attendence/attendance{date}.csv')
            cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50 , 50, 255), 1)
            attendance = [str(output[0]), str(date), str(timestamp)]
            

        cv2.imshow("Video", frame)
        k = cv2.waitKey(1)

        if k == ord('o'):
            st.info("Attendance Taken")
            time.sleep(5)
            if exist:
                pass
            else:
                if not os.path.exists(f'Attendence/{branch}_{year}_'):
                    os.makedirs(f'Attendence/{branch}_{year}_')
                with open(f'Attendence/{branch}_{year}_/attendance_' + date + '.csv', '+a') as csvfile:
                    writer = csv.writer(csvfile)
                    if not os.path.getsize(f'Attendence/{branch}_{year}_/attendance_' + date + '.csv'):
                        writer.writerow(COL_NAMES)
                    writer.writerow(attendance)


        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def view_attendance_data(branch, year, date):
    st.title("View Attendance Data")

    # Get the formatted date
    formatted_date = date.strftime("%d-%m-%Y")

    try:
        # Read the attendance data from the CSV file
        filename = f'Attendence/{branch}_{year}_/attendance_{formatted_date}.csv'
        df = pd.read_csv(filename)
        st.table(df)

        # Add a button to download the CSV file
        csv_data = df.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        button_label = "Download CSV"
        button_uuid = str(uuid.uuid4()).replace("-", "")
        button_id = f"button_{button_uuid}"
        # if st.button(button_label, key=button_id):
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="attendance_{formatted_date}.csv">Click here to download</a>',
            unsafe_allow_html=True
        )

    except FileNotFoundError:
        st.error("Attendance data not found for the selected date.")

def view_todays_attendance(branch, year):
    st.title("View Today's Attendance")
    
    formatteds_date = datetime.now().strftime("%d-%m-%Y")
    try:
        filename = f'Attendence/{branch}_{year}_/attendance_{formatteds_date}.csv'
        df = pd.read_csv(filename)
        st.table(df)
    except FileNotFoundError:

    # View today's attendance (same as your existing function)
      pass

def main():
    st.title("Attendance System")

    option = st.sidebar.selectbox("Select Option", ["Home","Register Faces", "Take Attendance", "View Attendance Data", "View Today's Attendance"])
    
    if option == "Home":
        home()

    elif option == "Register Faces":
        register_faces()

    elif option == "Take Attendance":
        branch = st.sidebar.selectbox("Select Branch", ['CSE', 'ME', 'EX', 'EC', 'CE', 'AIADS'])
        year = st.sidebar.selectbox("Select Year", ['1', '2', '3', '4'])
        take_attendance(branch, year)

    elif option == "View Attendance Data":
        branch = st.sidebar.selectbox("Select Branch", ['CSE', 'ME', 'EX', 'EC', 'CE', 'AIADS'])
        year = st.sidebar.selectbox("Select Year", ['1', '2', '3', '4'])
        date = st.sidebar.date_input("Enter Date")
        view_attendance_data(branch, year, date)

    elif option == "View Today's Attendance":
        branch = st.sidebar.selectbox("Select Branch", ['CSE', 'ME', 'EX', 'EC', 'CE', 'AIADS'])
        year = st.sidebar.selectbox("Select Year", ['1', '2', '3', '4'])
        view_todays_attendance(branch, year)
        
if __name__ == "__main__":
    main()
