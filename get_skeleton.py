import cv2
import mediapipe as mp
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def give_landmarks(video_path, label, fps):

    body_pose_landmarks = ["nose", "left eye inner", "left eye", "left eye outer",
                           "right eye inner", "right eye", "right eye outer",
                           "left ear", "right ear", "mouth left", "mouth right",
                           "left shoulder", "right shoulder", "left elbow",
                           "right elbow", "left wrist", "right wrist", "left pinky",
                           "right pinky", "left index", "right index", "left thumb",
                           "right thumb", "left hip", "right hip", "left knee",
                           "right knee", "left ankle", "right ankle", "left heel",
                           "right heel", "left foot index", "right foot index"]

    col_name = []
    for i in body_pose_landmarks:
        col_name += [i + '_X', i + '_Y', i + '_Z', i + '_V']

    frame_count = 0
    df_list = []

    cap = cv2.VideoCapture(video_path)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    landmark_mp_list = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # No more frames in the video

            # Skip frames according to desired FPS
            if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) != 0:
                frame_count += 1
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            landmark_mp_list.append(results)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                print('No landmarks found')
            frame_count += 1

            if landmarks is not None:
                pose_row = []
                for landmark in landmarks:
                    pose_row += [landmark.x, landmark.y,
                                 landmark.z, landmark.visibility]

                # append the pose row to the dataframe
                df_list.append(pd.DataFrame([pose_row], columns=col_name))

    cap.release()
    df = pd.concat(df_list, ignore_index=True)
    df['label'] = label  # Add label column

    return df, landmark_mp_list


def make_skeleton_frame(landmarks):

    if landmarks is None:
        return

    LANDMARK_GROUPS = [
        [8, 6, 5, 4, 0, 1, 2, 3, 7],   # eyes
        [10, 9],                       # mouth
        [11, 13, 15, 17, 19, 15, 21],  # right arm
        [11, 23, 25, 27, 29, 31, 27],  # right body side
        [12, 14, 16, 18, 20, 16, 22],  # left arm
        [12, 24, 26, 28, 30, 32, 28],  # left body side
        [11, 12],                      # shoulder
        [23, 24],                      # waist
    ]

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    
    # fig with 400 x 300 pixels
    fig, ax = plt.subplots(figsize=(4, 3))
    # ax.invert_xaxis()
    ax.invert_yaxis()
    # ax.set_xlim3d(-1, 1)
    # ax.set_ylim3d(-1, 1)
    # ax.set_zlim3d(1, -1)
    ax.set_aspect('equal')
    ax.axis('off')

    for group in LANDMARK_GROUPS:

        plotX, plotY, plotZ = [], [], []
        plotX = [landmarks.landmark[i].x for i in group]
        plotY = [landmarks.landmark[i].y for i in group]
        # plotZ = [landmarks.landmark[i].z for i in group]

        ax.plot(plotX, plotY, marker='o',
                markersize=5, linestyle='-', color='r', lw=5)

    return fig


# video_path = 'false_1.mp4'
# label = 'chair_pose'
# fps = 30
# landmarks, landmark_mp_list = give_landmarks(video_path, label, fps)
# fig = make_skeleton_frame(landmark_mp_list[70].pose_landmarks)

# plt.show()
