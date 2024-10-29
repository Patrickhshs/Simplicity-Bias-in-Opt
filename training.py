import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import load_cifar10
import matplotlib.pyplot as plt
import json
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers=2, num_filters=[6, 16], num_fc_layers=3, num_neurons=[120, 84, 10]):
        super(SimpleCNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.num_fc_layers = num_fc_layers
        self.num_neurons = num_neurons

        self.convs = nn.ModuleList()
        in_channels = 3
        for i in range(num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels, num_filters[i], 5))
            in_channels = num_filters[i]

        self.pool = nn.MaxPool2d(2, 2)
        self.fcs = nn.ModuleList()
        in_features = num_filters[-1] * 5 * 5
        for i in range(num_fc_layers):
            self.fcs.append(nn.Linear(in_features, num_neurons[i]))
            in_features = num_neurons[i]

    def forward(self, x):
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))
        x = x.view(-1, self.num_filters[-1] * 5 * 5)
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)
        return x

def train_model(num_conv_layers=2, num_filters=[6, 16], num_fc_layers=3, num_neurons=[120, 84, 10], num_epochs=100, save_dir='./results'):
    trainloader, testloader = load_cifar10()
    net = SimpleCNN(num_conv_layers=num_conv_layers, num_filters=num_filters, num_fc_layers=num_fc_layers, num_neurons=num_neurons)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        test_accuracy = evaluate_model(net, testloader)
        test_accuracies.append(test_accuracy)

        print(f'[Epoch {epoch + 1}] loss: {running_loss / len(trainloader):.3f}, train accuracy: {train_accuracy:.2f}%, test accuracy: {test_accuracy:.2f}%')
        if running_loss < 0.001:
            break
    
    print('Finished Training')
    plot_accuracies(train_accuracies, test_accuracies, num_fc_layers, num_neurons, save_dir)
    save_evaluation_data(train_accuracies, test_accuracies, num_fc_layers, num_neurons, save_dir)
    return net, testloader

def evaluate_model(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def plot_accuracies(train_accuracies, test_accuracies, num_fc_layers, num_neurons, save_dir):
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Train and Test Accuracy over Epochs\nFC Layers: {num_fc_layers}, Neurons: {num_neurons}')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'accuracy_plot_fc_layers_{num_fc_layers}_neurons_{num_neurons}.png'))
    plt.show()

def save_evaluation_data(train_accuracies, test_accuracies, num_fc_layers, num_neurons, save_dir):
    evaluation_data = {
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'num_fc_layers': num_fc_layers,
        'num_neurons': num_neurons
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'evaluation_data_fc_layers_{num_fc_layers}_neurons_{num_neurons}.json'), 'w') as f:
        json.dump(evaluation_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a SimpleCNN on CIFAR-10')
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of convolutional layers')
    parser.add_argument('--num_filters', type=str, default='6,16', help='Comma-separated list of number of filters for each conv layer')
    parser.add_argument('--num_fc_layers', type=str, default='3', help='Number of fully connected layers')
    parser.add_argument('--num_neurons', type=str, default='120,84,10', help='Comma-separated list of number of neurons for each FC layer')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save evaluation data and plots')

    args = parser.parse_args()

    num_filters = list(map(int, args.num_filters.split(',')))
    num_fc_layers = int(args.num_fc_layers)
    num_neurons = list(map(int, args.num_neurons.split(',')))

    net, testloader = train_model(num_conv_layers=args.num_conv_layers, num_filters=num_filters, num_fc_layers=num_fc_layers, num_neurons=num_neurons, num_epochs=args.num_epochs, save_dir=args.save_dir)