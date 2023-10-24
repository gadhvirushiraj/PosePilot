import cv2
import math
import json
import torch
from torch import nn
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
import argparse
from model import SimpleNN
from itertools import combinations
import numpy as np

def get_landmarks(img_pth, check_landmarks):

    # load mediapipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    custom_style = mp.solutions.drawing_styles.DrawingSpec(
        color=(0, 255, 0),  # Color of keypoints
        thickness=12,        # Thickness of keypoints
        circle_radius=7,    # Radius of keypoints
    )
    
    connection_style = mp.solutions.drawing_styles.DrawingSpec(
        color=(255, 255, 255),  # Color of connection lines
        thickness=5,        # Increase the thickness for connection lines
    )

    # load image
    img = cv2.imread(img_pth)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = []
    # Create a Pose object
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Process the image to detect body pose
        results = pose.process(img)

        if results.pose_landmarks:
            # Draw the landmarks on the image
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, custom_style, connection_style)
            
            # Save the landmarks in a DataFrame
            for landmark in results.pose_landmarks.landmark:
                data += [landmark.x, landmark.y, landmark.z, landmark.visibility]

    if check_landmarks:
        plt.imshow(img)
        plt.title('Landmarks Detected')
        plt.show()

    return img, data


def angle(point1, point2, point3):
    """ Calculate angle between two lines """
    if(point1==(0,0) or point2==(0,0) or point3==(0,0)):
        return 0
    numerator = point2[1] * (point1[0] - point3[0]) + point1[1] * \
                (point3[0] - point2[0]) + point3[1] * (point2[0] - point1[0])
    denominator = (point2[0] - point1[0]) * (point1[0] - point3[0]) + \
                (point2[1] - point1[1]) * (point1[1] - point3[1])
    try:
        ang = math.atan(numerator/denominator)
        ang = ang * 180 / math.pi
        if ang < 0:
            ang = 180 + ang
        return ang
    except:
        return 90.0


def get_features(data):

    body_pose_landmarks = ["nose","left eye inner","left eye","left eye outer",
                        "right eye inner","right eye","right eye outer",
                        "left ear","right ear","mouth left","mouth right",
                        "left shoulder","right shoulder","left elbow",
                        "right elbow","left wrist","right wrist","left pinky",
                        "right pinky","left index","right index","left thumb",
                        "right thumb","left hip","right hip","left knee",
                        "right knee","left ankle","right ankle","left heel",
                        "right heel", "left foot index","right foot index"]
    col_name = []
    for i in body_pose_landmarks:
        col_name += [i + ' X', i + ' Y', i + ' Z', i + ' V']

    # rename columns
    landmark_df = pd.DataFrame(columns=col_name)
    landmark_df.loc[0] = data

    # read top_100_features_names.json
    top_features_names = []
    with open('top_100_features_names.json') as json_file:
        top_features_names = json.load(json_file)
    
    # calculate all the angles form the top feature names and add to top feature names
    feature_df = pd.DataFrame(columns=top_features_names.keys())
    for i in top_features_names.keys():
        pt1 = (landmark_df[top_features_names[i][0] + ' X'].values, landmark_df[top_features_names[i][0] + ' Y'].values)
        pt2 = (landmark_df[top_features_names[i][1] + ' X'].values, landmark_df[top_features_names[i][1] + ' Y'].values)
        pt3 = (landmark_df[top_features_names[i][2] + ' X'].values, landmark_df[top_features_names[i][2] + ' Y'].values)
        feature_df.loc[0, i] = angle(pt1, pt2, pt3)

    # Read scaler from scaler.npy
    scaler = np.load('scaler.npy', allow_pickle=True).item()  # Load the scaler as a dictionary

    # Apply scaler
    feature_df = scaler.transform(feature_df)

    # Read PCA components from pca_components.npy
    pca_components = np.load('pca_components.npy', allow_pickle=True)

    # Apply PCA
    feature_df = np.dot(feature_df, pca_components.T)

    return feature_df


def predict(model_path, print_prob):

    model = SimpleNN(feature_df.shape[1])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    input_data = torch.tensor(feature_df, dtype=torch.float32)

    # reading mapping.json file for target names
    with open('mapping.json') as json_file:
        target_names = json.load(json_file)

    # Make predictions
    with torch.no_grad():
        pred = model(input_data)
        if print_prob:
            print(pred)

    # Get predicted class
    pred_idx = torch.argmax(pred, dim=1)
    print('\nPredicted Yoga Pose: ',target_names[str(int(pred_idx))])



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Predict Yoga Pose')
    parser.add_argument('--model', type=str, default='model.pth', help='path to model')
    parser.add_argument('--image', type=str, help='path to image')
    parser.add_argument('--check_landmarks', action='store_true', help='display landmark detected image')
    parser.add_argument('--print_prob', action='store_true', help='print probability of each class')

    args = parser.parse_args()
    img, data = get_landmarks(args.image, args.check_landmarks)
    feature_df = get_features(data)
    predict(args.model, args.print_prob)