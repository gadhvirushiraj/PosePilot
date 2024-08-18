import numpy as np
import pandas as pd
import pickle

from scipy.signal import find_peaks

from correction_model import RNN

import matplotlib.pyplot as plt

import torch

from utils import structure_data,cal_error, correction_angles_convert


def scale_data(data_input, scalers):

    
    for i in range(len(scalers)):
        data_input.iloc[:, i] = scalers[i].transform(data_input.iloc[:, i].values.reshape(-1, 1))
    return data_input

def prepare_data(data_input, desired_peak_count=20):
    df_peaks_1 = pd.DataFrame()
    for k in range(1):
        peaks_1 = np.linspace(k*85+1, (k+1)*len(data_input[k*85:(k+1)*85])-2, num=desired_peak_count, dtype=int)
        df_peaks_1 = pd.concat([df_peaks_1, data_input.iloc[peaks_1]])
    return df_peaks_1.reset_index(drop=True), peaks_1

def load_model(device, input_size, hidden_size, num_layers, num_output_features,pose):
    model = RNN(input_size, hidden_size, num_layers, num_output_features).to(device)
    model.load_state_dict(torch.load(f"correction_chair.pth"))
    model.eval()
    return model

def test(all_time_data, device,pose):
    data_input, peaks_1 = prepare_data(all_time_data)
    data_original = all_time_data.copy()
    scalers = pickle.load(open('scalers.pkl', 'rb'))
    data_input = scale_data(data_input, scalers)

    window_size = 1
    input_size = 9 * window_size
    hidden_size = 128
    num_layers = 1
    num_output_features = 9

    columns_to_drop = ['label', 'error']
    data_input = data_input.drop(columns=columns_to_drop, axis=1)

    model = load_model(device, input_size, hidden_size, num_layers, num_output_features,pose)
    data_input_tensor = torch.tensor(data_input.values, dtype=torch.float32).to(device)
    outputs, _ = model(data_input_tensor)

    outputs = outputs.cpu().detach().numpy()
    for i in range(len(scalers)):
        outputs[:, i] = scalers[i].inverse_transform(outputs[:, i].reshape(-1, 1)).reshape(-1)

    transformed_data_input = data_input_tensor.cpu().detach().numpy()
    for i in range(len(scalers)):
        transformed_data_input[:, i] = scalers[i].inverse_transform(transformed_data_input[:, i].reshape(-1, 1)).reshape(-1)

    plot_results(data_original, transformed_data_input, outputs,peaks_1)


def plot_results(data_original, data_input, outputs,peaks_1):
    feature_label = [
        'Left Elbow', 'Right Elbow', 'Left Hip', 'Right Hip', 'Left Knee',
        'Right Knee', 'Neck', 'Left Shoulder', 'Right Shoulder'
    ]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        for j in range(3):
            axs[i, j].set_xlim([0, len(data_original)])
            axs[i, j].plot(data_original.iloc[:, 3*i+j], label='Original', linestyle='--')
            axs[i, j].plot(peaks_1, data_input[:, 3*i+j], label='Input', linestyle='', marker='o')
            axs[i, j].plot(peaks_1, outputs[:, 3*i+j], label='Output', linestyle='-')
            lw_1std = outputs[:, 3*i+j] - np.std(outputs[:, 3*i+j])
            up_1std = outputs[:, 3*i+j] + np.std(outputs[:, 3*i+j])
            axs[i, j].fill_between(peaks_1, lw_1std, up_1std, alpha=0.2, color='grey', label='1-std band')
            lw_2std = outputs[:, 3*i+j] - 2*np.std(outputs[:, 3*i+j])
            up_2std = outputs[:, 3*i+j] + 2*np.std(outputs[:, 3*i+j])
            axs[i, j].fill_between(peaks_1, lw_2std, up_2std, alpha=0.1, color='grey', label='1.5-std band')

            axs[i, j].set_title('Feature ' + str(3*i+j+1) + " " + f"({feature_label[3*i+j]})", fontsize=15)

            for k in range(len(peaks_1)):
                if data_input[k, 3*i+j] < lw_1std[k]:
                    axs[i, j].plot(peaks_1[k], data_input[k, 3*i+j], 'ro', label='incorrect points')
                    axs[i, j].arrow(peaks_1[k], data_input[k, 3*i+j], 0, lw_1std[k]-data_input[k, 3*i+j], head_width=2, head_length=0.1, fc='red', ec='red', label='correction vector')
                if data_input[k, 3*i+j] > up_1std[k]:
                    axs[i, j].plot(peaks_1[k], data_input[k, 3*i+j], 'ro', label='incorrect points')
                    axs[i, j].arrow(peaks_1[k], data_input[k, 3*i+j], 0, up_1std[k]-data_input[k, 3*i+j], head_width=2, head_length=0.1, fc='red', ec='red', label='correction vector')

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique_labels = sorted(list(set(labels)))
    selected_lines = [lines[labels.index(label)] for label in unique_labels]

    fig.subplots_adjust(top=0.2)
    fig.legend(selected_lines, unique_labels, loc='upper center', framealpha=1, fontsize=13, ncol=7, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('YogaPal', fontsize=20, y=1.05)
    fig.supxlabel('Frames (n)', fontsize=15, x=0.5, y=0.004)
    fig.supylabel('Degrees (°)', fontsize=15, y=0.5, x=0.004)
    plt.tight_layout()
    plt.savefig('correction_output.png')
    plt.show()




def corr_predict(pose, data):
    data = structure_data(data)
    data = correction_angles_convert(data)
    data = cal_error(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test(data, device='cuda' if torch.cuda.is_available() else 'cpu',pose= pose)