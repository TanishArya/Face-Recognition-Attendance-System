## Face Recognition Attendance System

This Python script implements a Face Recognition Attendance System using the Streamlit library for creating the user interface, OpenCV for capturing and processing images, and scikit-learn for building and using the K-Nearest Neighbors classifier for face recognition.

#### Here's a breakdown of the functionality:

##### 1.Imports:
  Import necessary libraries such as Streamlit, OpenCV, pickle, numpy, os, csv, time, datetime, pandas, base64, uuid, and the win32com.client module for text-to-speech functionality.

##### 2.Function Definitions:

- home(): Displays information about the Face Recognition Attendance System.
- register_faces(): Allows users to register their faces along with personal details such as name, enrollment number, branch, and year.
- take_attendance(branch, year): Captures faces through the camera and recognizes them to mark attendance.
- view_attendance_data(branch, year, date): Allows users to view attendance data for a specific date.
- view_todays_attendance(branch, year): Displays today's attendance for a selected branch.

##### 3.Main Functionality:

- main(): Provides the main user interface using Streamlit. It includes options for home, registering faces, taking attendance, viewing attendance data, and viewing today's attendance. Users can select options from the sidebar.

##### 4.User Interface:

- The Streamlit library is used to create a user-friendly interface where users can interact with the system.
- Input fields, select boxes, and buttons are provided for user input and interaction.
- Output is displayed using Streamlit's text, write, table, and success functions.

##### 5.Face Recognition and Attendance:

- The script utilizes OpenCV to capture images from the camera and detect faces using a Haar Cascade classifier.
- It then uses a K-Nearest Neighbors classifier from scikit-learn to recognize faces based on previously registered faces.
- Attendance is marked by associating recognized faces with the current date and time and saving the data to CSV files.

##### 6.Data Management:

- Data such as names, enrollment numbers, faces, and attendance records are stored using pickle serialization and CSV files.
- Overall, this script provides a comprehensive solution for implementing a Face Recognition Attendance System with user-friendly interaction and efficient data management.

### 1. Overview

The Face Recognition Attendance System is a biometric technology that uses artificial intelligence to identify and verify individuals based on their facial characteristics. It captures images of faces using a digital camera, processes and analyzes them using computer vision techniques, and records attendance based on the recognized faces.

### 2. Features

- Face registration: Allows users to register their faces along with their personal details such as name, enrollment number, branch, and year.
- Attendance taking: Automatically takes attendance by recognizing faces captured by the camera.
- View attendance data: Allows users to view attendance data for specific dates or branches.
- View today's attendance: Provides a quick overview of today's attendance for a selected branch.

### 3. Installation

To run the Face Recognition Attendance System, follow these steps:

- Clone the repository:
  
```bash
git clone https://github.com/TanishArya/Face-Recognition-Attendance-System.git
```
  
- Install the required dependencies:

```bash
pip install -r requirements.txt
```

- Run the main script:

```bash
streamlit run main.py
```
### 4. Dependencies

##### Streamlit:
   Streamlit is a Python library used for building interactive web applications. It's particularly useful for creating data-driven applications quickly and easily.
##### OpenCV (cv2):
   OpenCV is a popular open-source computer vision library. In this code, it's used for capturing images from the camera, processing images (e.g., converting to grayscale, detecting faces), and displaying video frames.
##### Pickle:
   Pickle is a Python module used for serializing and deserializing Python objects. It's used here to save and load data such as names and faces to/from files.
##### NumPy (np): 
NumPy is a fundamental package for scientific computing with Python. It's used for array processing, reshaping images, and various numerical computations.
##### OS: 
The OS module provides functions for interacting with the operating system. It's used for tasks like file and directory manipulation, checking file existence, and creating directories.
##### CSV:
The CSV module provides functionality to read and write CSV files. It's used for handling attendance data, both for writing new attendance records and reading existing records.
##### Time:
The Time module provides various time-related functions. It's used for timestamping attendance records with the current date and time.
##### Datetime:
The Datetime module provides classes for manipulating dates and times. It's used for formatting dates and times in a human-readable format.
##### Scikit-learn: 
Scikit-learn is a machine learning library for Python. In this code, it's used for implementing the K-Nearest Neighbors classifier for face recognition.
##### win32com.client.Dispatch:
This module is part of the pywin32 package and is used for enabling text-to-speech functionality. It's used to provide auditory feedback when attendance is marked.
##### Pandas (pd):
Pandas is a powerful data analysis and manipulation library. It's used for reading CSV files into dataframes and displaying attendance data in tabular form.
##### Base64:
Base64 is a group of binary-to-text encoding schemes. It's used here for encoding CSV data before embedding it into HTML for download.
##### UUID:
UUID (Universally Unique Identifier) is a Python module for generating unique identifiers. It's used to generate unique IDs for HTML elements when embedding download links.
