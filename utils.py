"""
This module contains utility functions for the project.
"""

import numpy as np
from scipy.signal import find_peaks


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
