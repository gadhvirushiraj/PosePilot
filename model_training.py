import torch 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
from model import SimpleNN
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def load_data(feature_class, batch_size):
    train_df, test_df = train_test_split(feature_class, test_size=0.1, random_state=42)

    # making train data loader
    feature_torch = torch.tensor(train_df.drop('label',axis = 1).values, dtype=torch.float32)
    label_torch = torch.tensor(train_df['label'].values,dtype=torch.int64)
    train_dataset = TensorDataset(feature_torch, label_torch)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)   

    feature_torch_test = torch.tensor(test_df.drop('label',axis = 1).values, dtype=torch.float32)
    label_torch_test = torch.tensor(test_df['label'].values,dtype=torch.int64)
    test_dataset = TensorDataset(feature_torch_test, label_torch_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
 
def train(model, mapping,  epochs, lr, train_loader, test_loader, gpu):

    # define loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimize = optim.Adam(model.parameters(), lr=lr)

    # train on gpu if available and specified by user
    device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'
    running_loss = 0

    # store the loss and accuracy
    training_loss_list = []
    valid_acc_list = []

    # for confusion matrix
    y_pred = []
    y_test = []

    for epoch in range(epochs):
        for _, (features, labels) in enumerate(train_loader):
            
            # put data into gpu if available
            features, labels = features.to(device), labels.to(device)

            optimize.zero_grad()
            outputs = model.forward(features)
            loss_value = loss(outputs, labels)
            loss_value.backward()
            optimize.step()
            running_loss += loss_value.item()
            labels_correct = torch.argmax(outputs, dim=1) == labels

        if_last = True if epoch == (epochs-1) else False 

        # set model to evaluation mode
        model.eval()
        valid_acc = validate(model, y_pred, y_test, last = if_last)
        model.train()

        print(f'Epoch: {epoch+1}/{epochs}..',
            f'Training Loss: {running_loss/len(test_loader):.3f}',
            f'Training Accuracy: {100*len(labels_correct)/len(labels):.3f}%',
            f'Validation Accuracy: {valid_acc:.3f}%')

        training_loss_list.append(running_loss/len(test_loader))
        valid_acc_list.append(valid_acc)

        running_loss = 0.0

    # print the classification report
    y_pred = list(np.concatenate(y_pred))

    with open(mapping) as f:
        mapping = json.load(f)

    target_names = list(mapping.values())
    print('\n', classification_report(y_test, y_pred, target_names=target_names))

    return training_loss_list, valid_acc_list, y_pred, y_test, model


def validate(model, y_pred, y_test, last = False,):

    # train on gpu if available and specified by user
    device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'

    correct = 0
    total = 0
    for features, labels in test_loader: 
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)

        if last:
            y_pred.append(predicted.cpu().numpy())
            y_test.extend(labels.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_stats(training_loss_list, valid_acc_list, y_pred, y_test):

    # plot the training loss and validation accuracy in seperate plots next to each other
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    ax1.plot(training_loss_list, color='red')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(valid_acc_list, color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.suptitle('Model Training Statistics')
    plt.show()

    # plot the confusion matrix
    confusion = confusion_matrix(y_test,y_pred)
    disp = ConfusionMatrixDisplay(confusion, display_labels = ['tree','cobra','downdog','goddess','warrior','chair'])
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()



if __name__ == '__main__' :

    '''
    DEFAULT VALUES FOR TRAINING
        -- data_path = 'pca_features.csv'
        -- epochs = 200
        -- batch_size = 12
        -- lr = 0.001
        -- save_path = 'model.pth'
        -- gpu = 0
    '''

    # get the arguments from the user
    parser = argparse.ArgumentParser(description='Train Model for Yoga Pose Classification')
    data_path = parser.add_argument('--data_path', type=str, default='pca_features.csv', help='path to model')
    mapping = parser.add_argument('--mapping', type=str, default='mapping.json', help='path to mapping file')
    epochs = parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    batch_size = parser.add_argument('--batch_size', type=int, default=12, help='batch size for training and testing')
    lr = parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    save_path = parser.add_argument('--save_path', type=str, default='model.pth', help='path to save model')
    gpu = parser.add_argument('--gpu', type=int, default=0, help='use gpu for training')
    args = parser.parse_args()

    # read data
    feature_class = pd.read_csv(args.data_path)

    # calling respective functions
    model = SimpleNN(feature_class.shape[1]-1)
    train_loader, test_loader = load_data(feature_class, args.batch_size)
    training_loss_list, valid_acc_list, y_pred, y_test, model = train(model, args.mapping, args.epochs, args.lr, train_loader, test_loader, args.gpu)
    get_stats(training_loss_list, valid_acc_list, y_pred, y_test)

    # save the model
    torch.save(model.state_dict(), args.save_path) 