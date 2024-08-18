"""
This module contains utility functions for the project.
"""

import scipy.interpolate

import warnings
warnings.filterwarnings('ignore')


import numpy as np
from scipy.signal import find_peaks

import pandas as pd




def cal_angle(point1, point2, point3):
    """
    Calculate the angle between three points

    Parameters
    ----------
    point1 : tuple
        (x, y) coordinates of the first point

    point2 : tuple
        (x, y) coordinates of the second point

    point3 : tuple
        (x, y) coordinates of the third point

    Returns
    -------
    angle : float
        The angle between the three points in degrees
    """

    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def cal_error(data):
    """
    Calculate the error values for each frame.
    Error: Calculate the standard deviation of each window of 5 frames.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the angle values.\
        
    Returns
    -------
    data : DataFrame
        The original DataFrame with an additional column for the error values.
    """

    error_list = []
    for idx, _ in enumerate(data):
        temp = []
        for col in data.columns:
            window = data[col].loc[max(idx - 3, 0) : min(idx + 2, len(data))]
            temp.append(np.std(window.values))
        error_list.append(np.mean(temp))

    data["error"] = error_list
    return data


def selected_top_frames(data, desired_frames=10):
    """
    select the top number of desired frames based on the error values

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the angle values.

    desired_frames : int
        The number of frames to select.

    Returns
    -------
    data : DataFrame
        The DataFrame containing the selected frames.
    """

    min_prominence = 0.1
    error_values = data["error"]
    peaks, properties = find_peaks(error_values, distance=3, prominence=min_prominence)

    if len(peaks) < desired_frames:
        # insert elements btw most distant peaks, do till we have 10 peaks
        while len(peaks) < 10:
            diff = np.diff(peaks)
            max_diff = np.argmax(diff)
            peaks = np.insert(
                peaks, max_diff + 1, np.mean(peaks[max_diff : max_diff + 2])
            )
        print(
            f"WARNING: Peaks detected at prominence:{min_prominence} were less than desired frames."
        )
    else:
        # take peaks with highest prominences
        peaks = peaks[np.argsort(properties["prominences"])[-desired_frames:]]

    data = data.iloc[peaks].reset_index(drop=True)
    return data


def structure_data(data):
    """
    Structure the data and add column names, remove not needed data.
    """

    body_pose_landmarks = [
        "nose",
        "left eye inner",
        "left eye",
        "left eye outer",
        "right eye inner",
        "right eye",
        "right eye outer",
        "left ear",
        "right ear",
        "mouth left",
        "mouth right",
        "left shoulder",
        "right shoulder",
        "left elbow",
        "right elbow",
        "left wrist",
        "right wrist",
        "left pinky",
        "right pinky",
        "left index",
        "right index",
        "left thumb",
        "right thumb",
        "left hip",
        "right hip",
        "left knee",
        "right knee",
        "left ankle",
        "right ankle",
        "left heel",
        "right heel",
        "left foot index",
        "right foot index",
    ]

    col_name = []
    for i in body_pose_landmarks:
        col_name += [i + "_X", i + "_Y", i + "_Z", i + "_V"]

    data.columns = col_name
    data = data[data.columns[~data.columns.str.contains(" V")]]

    return data, body_pose_landmarks

def update_body_pose_landmarks(data,body_pose_landmarks)
    # remove certain body landmarks
    remove_list = [
        "left eye",
        "left eye inner",
        "left eye outer",
        "left ear",
        "right eye",
        "right eye inner",
        "right eye outer",
        "right ear",
        "mouth left",
        "mouth right",
        "left pinky",
        "right pinky",
        "left thumb",
        "right thumb",
        "left heel",
        "right heel",
    ]

    for i in remove_list:
        body_pose_landmarks.remove(i)
        data = data[data.columns[~data.columns.str.contains(i)]]

    return data, body_pose_landmarks


def correction_angles_convert(final_df):
#     f1: angle between left shoulder, left elbow, left wrist
#     f2: angle between right shoulder, right elbow, right wrist
#     f3: angle between left shoulder, left hip, left knee
#     f4: angle between right shoulder, right hip, right knee
#     f5: angle between left hip, left knee, left ankle
#     f6: angle between right hip, right knee, right ankle
#     f7: angle between nose, left shoulder, left hip
#     f8: angle between nose, right shoulder, right hip
#     f9: angle between left shoulder, nose, right shoulder
#     f10: angle between left knee, left ankle, left foot index
#     f11: angle between right knee, right ankle, right foot index
#     f12: angle between left index, left wrist, left thumb
#     f13: angle between right index, right wrist, right thumb
#     f14: angle between left shoulder, left hip, left foot index
#     f15: angle between right shoulder, right hip, right foot index

    feature_df = pd.DataFrame()

    feature_df['f1'] = final_df.apply(lambda x: cal_angle(x, (x['left shoulder_X'], x['left shoulder_Y']), (x['left elbow_X'], x['left elbow_Y']), (x['left wrist_X'], x['left wrist_Y'])), axis=1)
    feature_df['f2'] = final_df.apply(lambda x: cal_angle(x, (x['right shoulder_X'], x['right shoulder_Y']), (x['right elbow_X'], x['right elbow_Y']), (x['right wrist_X'], x['right wrist_Y'])), axis=1)
    feature_df['f3'] = final_df.apply(lambda x: cal_angle(x, (x['left shoulder_X'], x['left shoulder_Y']), (x['left hip_X'], x['left hip_Y']), (x['left knee_X'], x['left knee_Y'])), axis=1)
    feature_df['f4'] = final_df.apply(lambda x: cal_angle(x, (x['right shoulder_X'], x['right shoulder_Y']), (x['right hip_X'], x['right hip_Y']), (x['right knee_X'], x['right knee_Y'])), axis=1)
    feature_df['f5'] = final_df.apply(lambda x: cal_angle(x, (x['left hip_X'], x['left hip_Y']), (x['left knee_X'], x['left knee_Y']), (x['left ankle_X'], x['left ankle_Y'])), axis=1)
    feature_df['f6'] = final_df.apply(lambda x: cal_angle(x, (x['right hip_X'], x['right hip_Y']), (x['right knee_X'], x['right knee_Y']), (x['right ankle_X'], x['right ankle_Y'])), axis=1)
    feature_df['f7'] = final_df.apply(lambda x: cal_angle(x, (x['left shoulder_X'], x['left shoulder_Y']), (x['nose_X'], x['nose_Y']), (x['right shoulder_X'], x['right shoulder_Y'])), axis=1)
    feature_df['f8'] = final_df.apply(lambda x: cal_angle(x, (x['left elbow_X'], x['left elbow_Y']), (x['left shoulder_X'], x['left shoulder_Y']), (x['left hip_X'], x['left hip_Y'])), axis=1)
    feature_df['f9'] = final_df.apply(lambda x: cal_angle(x, (x['right elbow_X'], x['right elbow_Y']), (x['right shoulder_X'], x['right shoulder_Y']), (x['right hip_X'], x['right hip_Y'])), axis=1)

    return feature_df



def reduce_rows(df,limit):
    """
    Reduces the number of rows in the DataFrame to the target length by removing rows with the smallest error.

    Parameters:
    target_length (int): The desired number of rows in the DataFrame.

    Returns:
    The DataFrame reduced to the target length.
    """

    target_length = limit
    
    while len(df) > target_length:
        # Find the index of the row with the minimum error
        min_error_index = df['error'].idxmin() 

        # Remove the row with the minimum error
        df = df.drop(min_error_index).reset_index(drop=True)

        # recalculate error
        df.drop('error', axis=1, inplace=True)
        df = cal_error(df)

    return df


def add_rows(df,limit):
    """
    Increases the number of rows in the DataFrame to the target length 
    by interpolating new rows at the highest error points.

    """

    target_length = limit

    while len(df) < target_length:
        # get the highest error index
        max_error_index = df['error'].idxmax()

        # decide +1 or -1 index to add; find difference between errors
        if np.sum(np.std(df.loc[max_error_index: max_error_index + 1][['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']])) < np.sum(np.std(df.loc[max_error_index - 1: max_error_index][['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']])):
            offset = -0.5
        else:
            offset = 0.5

        # Interpolate new values for each feature at the decided position
        features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']
        new_values = []
        for f in features:

            x = range(len(df))
            y = df[f]
            interp_point = min(max_error_index + offset, max(x))
            interp_value = scipy.interpolate.interp1d(x, y, kind='linear')(interp_point)
            new_values.append(interp_value)

        # Insert the interpolated row into the dataframe
        new_row = pd.DataFrame([new_values], columns=features)
        df = pd.concat([df.iloc[:max_error_index], new_row, df.iloc[max_error_index:]]).reset_index(drop=True)

        # Recalculate the errors
        df.drop('error', axis=1, inplace=True)
        df = cal_error(df)

    return df


def equal_rows(data,pose,limit):

    """
    Adjusts the number of rows in the input data to match the target length by either reducing or adding rows.

    """

    # Add rows if the current number of rows is less than the target length
    new = pd.DataFrame()
    if data.shape[0] < limit:
        new['label'] = data['label']
        data.drop(['label'],axis=1)
        data = add_rows(data,limit)
        data['label'] = new['label']

    

    # Reduce rows if the current number of rows is more than the target length
    if data.shape[0] > limit:
        new['label'] = data['label']
        data.drop(['label'],axis=1)
        data = reduce_rows(data,limit)
        data['label'] = new['label']


    return data