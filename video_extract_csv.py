import cv2 
import mediapipe as mp 
import numpy as np 
import csv 
import time
import os
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose= mp.solutions.pose


def export_lnd(landmarks ):
    if landmarks is not None:
    
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())
        

        with open('combined_new.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(pose_row) 



#Making CSV

def making_csv():
    Key_points = 33
    landmarks = []

    # Create headers for keypoints
    for i in range(1, Key_points + 1):
        landmarks += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

    # Write the headers to the CSV file
    with open('combined_new.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
def video_csv(video):
    making_csv()

    cap = cv2.VideoCapture(f"{video}")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    frame_count = 0
    start_time = time.time()

    list1 = []

    out = cv2.VideoWriter('video_website.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # No more frames in the video
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image) 


            list1.append(results)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            try:
                pose_re = results.pose_landmarks.landmark
            except:
                pass

            frame_count += 1

            export_lnd(pose_re)#file.split('_',1,)[1].split('.')[0])

            out.write(image)

        # break
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # fps = frame_count / elapsed_time


    # print(f"Frames of video {file} is : {frame_count}")
    # print(f"Elapsed Time of video {file} is : {elapsed_time:.2f} seconds")
    # print(f"FPS of video {file} is : {fps:.2f}")

    cap.release()