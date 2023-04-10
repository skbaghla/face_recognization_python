import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import glob
from pathlib import Path
import os
from face_recognition.face_recognition_cli import image_files_in_folder

if __name__ == '__main__':
    # capturing video
    video_capture = cv2.VideoCapture(0)
    # load known faces
    my_dir = '/Users/sanjeev/Documents/PythonLearningSanjeev/face_recognization/faces/'  # Folder where all your image files reside. Ensure it ends with '/
    known_faces_encodings = []  # Create an empty list for saving encoded files
    for i in os.listdir(my_dir):  # Loop over the folder to list individual files
        image = my_dir + i
        image = face_recognition.load_image_file(image)  # Run your load command
        image_encoding = face_recognition.face_encodings(image)  # Run your encoding command
        known_faces_encodings.append(image_encoding[0])  # Append the results to known_faces_encodings list

    # getting names of all files

    # known_faces_names = [os.path.basename(x) for x in glob.glob(my_dir)]
    known_faces_names = list(filter(lambda filepath: filepath.is_file(), Path(my_dir).glob('*')))

    # list of expected students
    students = known_faces_names.copy()

    # get current date and time
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    # creating CSV writer

    f = open(f"{current_date}.csv", "w+", newline="")
    lnwriter = csv.writer(f)

    while True:
        _, frame = video_capture.read()  # _ = was successful or not , frame = video frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Recognize faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matchess = face_recognition.compare_faces(known_faces_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_faces_encodings, face_encoding)
            best_matcg_index = np.argmin(face_distance)

            if matchess[best_matcg_index]:
                name = known_faces_names[best_matcg_index]
                print(name)
            # Add the Text if person is present

                if name in known_faces_names:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCorner = (10,100)
                    fontScale = 1.5
                    fontColor = (255,0,0)
                    thickness = 3
                    lineType = 2
                    shortname = os.path.basename(name)
                    cv2.putText(frame, shortname + " Present", bottomLeftCorner, font, fontScale, fontColor, thickness, lineType)

                    if name in students:
                        students.remove(name)
                        currentTime = now.strftime("%H:%M:%S")
                        shortname1 = os.path.basename(name)
                        print(f"{shortname1}  {name}")
                        lnwriter.writerow([shortname1 + " " + currentTime])

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    f.close()
