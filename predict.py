import cv2
import math
import torch
from torch import nn
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
import argparse
from model import SimpleNN


def get_landmarks(img_pth, check_landmarks):

    # load mediapipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
    custom_connections = list(mp_pose.POSE_CONNECTIONS)

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
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
            # Save the landmarks in a DataFrame
            for landmark in results.pose_landmarks.landmark:
                data += [landmark.x, landmark.y, landmark.z, landmark.visibility]

    if check_landmarks:
        plt.imshow(img)
        plt.title('Landmarks Detected')
        plt.show()

    return img, data

def angle(self, point1, point2, point3):
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
    final_df = pd.DataFrame(columns=col_name)
    final_df.loc[0] = data

    final_df = final_df[final_df.columns[~final_df.columns.str.contains(' V')]]

    '''
    f1: angle between left shoulder, left elbow, left wrist
    f2: angle between right shoulder, right elbow, right wrist
    f3: angle between left shoulder, left hip, left knee
    f4: angle between right shoulder, right hip, right knee
    f5: angle between left hip, left knee, left ankle
    f6: angle between right hip, right knee, right ankle
    f7: angle between nose, left shoulder, left hip
    f8: angle between nose, right shoulder, right hip
    f9: angle between left shoulder, nose, right shoulder
    f10: angle between left knee, left ankle, left foot index
    f11: angle between right knee, right ankle, right foot index
    f12: angle between left index, left wrist, left thumb
    f13: angle between right index, right wrist, right thumb
    f14: angle between left shoulder, left hip, left foot index
    f15: angle between right shoulder, right hip, right foot index
    '''

    feature_df = pd.DataFrame()

    feature_df['f1'] = final_df.apply(lambda x: angle(x, (x['left shoulder X'], x['left shoulder Y']), (x['left elbow X'], x['left elbow Y']), (x['left wrist X'], x['left wrist Y'])), axis=1)
    feature_df['f2'] = final_df.apply(lambda x: angle(x, (x['right shoulder X'], x['right shoulder Y']), (x['right elbow X'], x['right elbow Y']), (x['right wrist X'], x['right wrist Y'])), axis=1)
    feature_df['f3'] = final_df.apply(lambda x: angle(x, (x['left shoulder X'], x['left shoulder Y']), (x['left hip X'], x['left hip Y']), (x['left knee X'], x['left knee Y'])), axis=1)
    feature_df['f4'] = final_df.apply(lambda x: angle(x, (x['right shoulder X'], x['right shoulder Y']), (x['right hip X'], x['right hip Y']), (x['right knee X'], x['right knee Y'])), axis=1)
    feature_df['f5'] = final_df.apply(lambda x: angle(x, (x['left hip X'], x['left hip Y']), (x['left knee X'], x['left knee Y']), (x['left ankle X'], x['left ankle Y'])), axis=1)
    feature_df['f6'] = final_df.apply(lambda x: angle(x, (x['right hip X'], x['right hip Y']), (x['right knee X'], x['right knee Y']), (x['right ankle X'], x['right ankle Y'])), axis=1)
    feature_df['f7'] = final_df.apply(lambda x: angle(x, (x['nose X'], x['nose Y']), (x['left shoulder X'], x['left shoulder Y']), (x['left hip X'], x['left hip Y'])), axis=1)
    feature_df['f8'] = final_df.apply(lambda x: angle(x, (x['nose X'], x['nose Y']), (x['right shoulder X'], x['right shoulder Y']), (x['right hip X'], x['right hip Y'])), axis=1)
    feature_df['f9'] = final_df.apply(lambda x: angle(x, (x['left shoulder X'], x['left shoulder Y']), (x['nose X'], x['nose Y']), (x['right shoulder X'], x['right shoulder Y'])), axis=1)
    feature_df['f10'] = final_df.apply(lambda x: angle(x, (x['left knee X'], x['left knee Y']), (x['left ankle X'], x['left ankle Y']), (x['left foot index X'], x['left foot index Y'])), axis=1)
    feature_df['f11'] = final_df.apply(lambda x: angle(x, (x['right knee X'], x['right knee Y']), (x['right ankle X'], x['right ankle Y']), (x['right foot index X'], x['right foot index Y'])), axis=1)
    feature_df['f12'] = final_df.apply(lambda x: angle(x, (x['left index X'], x['left index Y']), (x['left wrist X'], x['left wrist Y']), (x['left thumb X'], x['left thumb Y'])), axis=1)
    feature_df['f13'] = final_df.apply(lambda x: angle(x, (x['right index X'], x['right index Y']), (x['right wrist X'], x['right wrist Y']), (x['right thumb X'], x['right thumb Y'])), axis=1)
    feature_df['f14'] = final_df.apply(lambda x: angle(x, (x['left shoulder X'], x['left shoulder Y']), (x['left hip X'], x['left hip Y']), (x['left foot index X'], x['left foot index Y'])), axis=1)
    feature_df['f15'] = final_df.apply(lambda x: angle(x, (x['right shoulder X'], x['right shoulder Y']), (x['right hip X'], x['right hip Y']), (x['right foot index X'], x['right foot index Y'])), axis=1)


    return feature_df

def predict(model_path, print_prob):

    model = SimpleNN(15)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    input_data = torch.tensor(feature_df.values, dtype=torch.float32)
    target_names = ['tree','cobra','downdog_data','goddess','warrior','chair']

    # Make predictions
    with torch.no_grad():
        pred = model(input_data)
        if print_prob:
            print(pred)

    # Get predicted class
    pred_idx = torch.argmax(pred, dim=1)
    print('\nYoga Pose: ',target_names[pred_idx])



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Predict Yoga Pose')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to model')
    parser.add_argument('--image', type=str, help='Path to image')
    parser.add_argument('--check_landmarks', action='store_true', help='Display landmark detected image')
    parser.add_argument('--print_prob', action='store_true', help='Print probability of each class')

    args = parser.parse_args()

    img, data = get_landmarks(args.image, args.check_landmarks)
    feature_df = get_features(data)
    predict(args.model, args.print_prob)