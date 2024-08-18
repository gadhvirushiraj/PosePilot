"""
Pipeline for predictions using the trained classifier.
"""

from itertools import combinations

import torch

from classify_model import ClassifyPose
from utils import structure_data, cal_angle, cal_error, selected_top_frames, update_body_pose_landmarks


def feature_classify(data):
    """
    Extract the features from the data for classifier.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the body pose landmarks.

    Returns
    -------
    feature_df : DataFrame
        The DataFrame containing the features for classification.
    """

    feature_df, body_pose_landmarks = structure_data(data)
    
    feature_df, body_pose_landmarks = update_body_pose_landmarks(feature_df,body_pose_landmarks)

    # generate a mapping for feature to angle reference
    mapping = {}
    all_angles = list(combinations(body_pose_landmarks, 3))

    for idx, _ in enumerate(all_angles):
        feature_df["f" + str(idx + 1)] = data.apply(
            lambda x: cal_angle(
                (x[all_angles[idx][0] + "_X"], x[all_angles[idx][0] + "_Y"]),
                (x[all_angles[idx][1] + "_X"], x[all_angles[idx][1] + "_Y"]),
                (x[all_angles[idx][2] + "_X"], x[all_angles[idx][2] + "_Y"]),
            ),
            axis=1,
        )
        mapping["f" + str(idx + 1)] = all_angles[idx]

    feature_df = cal_error(feature_df)
    feature_df = selected_top_frames(feature_df)

    return feature_df


def config_model(path="./models/classify-model.pth"):
    """
    Configure the classifier model.

    Returns
    -------
    model : ClassifyPose
        The classifier model.
    """

    input_size = 680
    hidden_size = 256
    num_layers = 1
    num_classes = 6
    sequence_length = 10

    model = ClassifyPose(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        sequence_length=sequence_length,
    )
    model.load_state_dict(torch.load(path))

    return model


def predict(data, model):
    """
    Predict the class of the input data.
    """

    feature_df = feature_classify(data)
    feature_df.drop(columns=["error", "label"], inplace=True)
    input_feature = torch.tensor(feature_df.values, dtype=torch.float32).to(device)
    input_feature = input_feature.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_feature)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()
