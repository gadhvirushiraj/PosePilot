import numpy as np
import pandas as pd
import pickle

from scipy.signal import find_peaks

from correction_model import RNN

import matplotlib.pyplot as plt

import torch

from utils import structure_data,cal_error, correction_angles_convert, equal_rows


def scale_data(data_input, scalers):
    '''
    scales the input data using the provided scalers

    iterates over each scaler and applies it to the corresponding column in the input data.

    '''

    for i, scaler in enumerate(scalers):
        data_input.iloc[:, i] = scaler.transform(data_input.iloc[:, i].values.reshape(-1, 1))
    return data_input

def prepare_data(data_input, desired_frame_count=20,seq_length):
    '''
    prepares the data by finding peaks in the input data.

    desired_frame_count = The number of peaks to select.

    '''
    frames_selected = np.linspace( 1, len(data_input[:seq_length]) - 2, num=desired_frame_count, dtype=int)
    data_input = data_input.iloc[frames_selected]
    return data_input.reset_index(drop=True), frames_selected

def load_model(device, input_size, hidden_size, num_layers, num_output_features):
    '''
    Loading the model which is stored locally and importing RNN model from the correction_model.py
    '''
    model = RNN(input_size, hidden_size, num_layers, num_output_features).to(device)
    model.load_state_dict(torch.load(f"correction_chair.pth"))
    model.eval()
    return model

def test(data_input, device,pose):
    # Prepare data by finding frames within the specified sequence length.

    data_input = equal_rows(data_input,pose,seq_length=85)
    data_input, frames_selected = prepare_data(data_input,seq_length=85)

    data_original = data_input.copy()
    # Load scalers to normalize the data.
    scalers = pickle.load(open('scalers.pkl', 'rb'))
    
    # Scale the input data using the loaded scalers.
    data_input = scale_data(data_input, scalers)

    # model parameters
    window_size = 1
    input_size = 9 * window_size
    hidden_size = 128
    num_layers = 1
    num_output_features = 9

    # cleaning the input data

    columns_to_drop = ['label', 'error']
    data_input = data_input.drop(columns=columns_to_drop, axis=1)

    model = load_model(device, input_size, hidden_size, num_layers, num_output_features,pose)
    data_input_tensor = torch.tensor(data_input.values, dtype=torch.float32).to(device)
    outputs, _ = model(data_input_tensor)

    #converting the model outputs into numpy array

    outputs = outputs.cpu().detach().numpy()
    for i in range(len(scalers)):
        outputs[:, i] = scalers[i].inverse_transform(outputs[:, i].reshape(-1, 1)).reshape(-1)

    # Reverse scaling on the transformed input data for comparison.

    transformed_data_input = data_input_tensor.cpu().detach().numpy()
    for i in range(len(scalers)):
        transformed_data_input[:, i] = scalers[i].inverse_transform(transformed_data_input[:, i].reshape(-1, 1)).reshape(-1)

    plot_results(data_original, transformed_data_input, outputs,frames_selected)


def plot_results(data_original, data_input, outputs,frames_selected):

    '''
    Plots the original data, input data, and model outputs for comparison, highlighting any 
    deviations with correction vectors.

    Returns:
    - Saves and displays the plot comparing original data, input data, and model outputs.

    '''


    feature_label = [
        'Left Elbow', 'Right Elbow', 'Left Hip', 'Right Hip', 'Left Knee',
        'Right Knee', 'Neck', 'Left Shoulder', 'Right Shoulder'
    ]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        for j in range(3):
            axs[i, j].set_xlim([0, len(data_original)])

            # Plot original, input, and output data for the current feature.
            axs[i, j].plot(data_original.iloc[:, 3*i+j], label='Original', linestyle='--')
            axs[i, j].plot(frames_selected, data_input[:, 3*i+j], label='Input', linestyle='', marker='o')
            axs[i, j].plot(frames_selected, outputs[:, 3*i+j], label='Output', linestyle='-')

            # Calculate and plot the 1-standard deviation band.
            lower_bound_1std = outputs[:, 3*i+j] - np.std(outputs[:, 3*i+j])
            upper_bound_1std = outputs[:, 3*i+j] + np.std(outputs[:, 3*i+j])
            axs[i, j].fill_between(frames_selected, lower_bound_1std, upper_bound_1std, alpha=0.2, color='grey', label='1-std band')

            # Calculate and plot the 2-standard deviation band.
            lower_bound_2std = outputs[:, 3*i+j] - 2*np.std(outputs[:, 3*i+j])
            upper_bound_2std = outputs[:, 3*i+j] + 2*np.std(outputs[:, 3*i+j])
            axs[i, j].fill_between(frames_selected, lower_bound_2std, upper_bound_2std, alpha=0.1, color='grey', label='1.5-std band')

            # Set the title for the subplot including feature name.
            axs[i, j].set_title('Feature ' + str(3*i+j+1) + " " + f"({feature_label[3*i+j]})", fontsize=15)

            # Highlight and correct points that fall outside the 1-standard deviation band.
            for k in range(len(frames_selected)):
                if data_input[k, 3*i+j] < lower_bound_1std[k]:
                    axs[i, j].plot(frames_selected[k], data_input[k, 3*i+j], 'ro', label='incorrect points')
                    axs[i, j].arrow(frames_selected[k], data_input[k, 3*i+j], 0, lower_bound_1std[k]-data_input[k, 3*i+j], head_width=2, head_length=0.1, fc='red', ec='red', label='correction vector')
                if data_input[k, 3*i+j] > upper_bound_1std[k]:
                    axs[i, j].plot(frames_selected[k], data_input[k, 3*i+j], 'ro', label='incorrect points')
                    axs[i, j].arrow(frames_selected[k], data_input[k, 3*i+j], 0, upper_bound_1std[k]-data_input[k, 3*i+j], head_width=2, head_length=0.1, fc='red', ec='red', label='correction vector')

    
    # combine and organize legends from all subplots
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique_labels = sorted(list(set(labels)))
    selected_lines = [lines[labels.index(label)] for label in unique_labels]


    #  Adjust layout and add titles, labels, and legend to the figure.
    fig.subplots_adjust(top=0.2)
    fig.legend(selected_lines, unique_labels, loc='upper center', framealpha=1, fontsize=13, ncol=7, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('PosePilot', fontsize=20, y=1.05)
    fig.supxlabel('Frames (n)', fontsize=15, x=0.5, y=0.004)
    fig.supylabel('Degrees (°)', fontsize=15, y=0.5, x=0.004)

    plt.tight_layout()
    plt.savefig('correction_output.png')
    plt.show()




def corr_predict(pose, data):

    data = structure_data(data)

    data = correction_angles_convert(data)

    data = cal_error(data)

    test(data, device='cuda' if torch.cuda.is_available() else 'cpu',pose= pose)