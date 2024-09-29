# importing libraries
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Model import CNNModel


def train_model(num_epochs):
    '''
     function for training the model

    :param num_epochs:
    :return:
    '''
    for epoch in range(num_epochs):
        # setting the model to training mode
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_sample = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            ''' forward pass'''
            outputs = model(images)
            loss = criterion(outputs, labels)
            '''backward propagation and optimizer'''
            # clear gradients
            optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # update model parameters
            optimizer.step()
            '''compute training loss and accuracy'''
            running_loss += loss.item()
            # predicted class
            _, predicted = torch.max(outputs, 1)
            # total number of images
            total_sample += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # per epoch loss and accuracy
        per_epoch_loss = running_loss / len(train_loader)
        per_epoch_acc = 100 * (correct_predictions / total_sample)

        # saving to history
        training_loss_history.append(per_epoch_loss)
        training_acc_history.append(per_epoch_acc)
        print(f"Epoch: [{epoch + 1}/{num_epochs}], Loss: {per_epoch_loss:.6f}, Accuracy: {per_epoch_acc:.3f}")

        # testing the model after each epoch
        test_accuracy = model_testing()
        testing_acc_history.append(test_accuracy)


def model_testing():
    '''
    Function for testing the model
    '''
    # setting the model into evaluation mode
    model.eval()
    correct_prediction = 0
    total_sample = 0
    # the below function won't track gradients
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_sample += labels.size(0)
            correct_prediction += (predicted == labels).sum().item()

    accuracy = 100 * (correct_prediction / total_sample)
    print(f"Testing Accuracy: {accuracy:.3f}%")
    return accuracy


def plotting_training_testing_result(num_epochs, training_loss, training_acc,
                                     testing_acc, learning_rate):
    '''
    Function to plot and save the training result graph
    :param num_epochs: Number of epochs
    :param training_loss: list of training losses per epoch
    :param training_acc: list of training accuracies per epoch
    :param testing_acc: list of testing accuracy per epoch
    :param learning_rate: learning_rate
    :return: none
    '''
    # create the plot
    plt.figure(figsize=(10, 5))
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss over Epochs for {learning_rate} lr', fontsize=10)
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), training_acc, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), testing_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy over Epochs for {learning_rate} lr', fontsize=10)
    plt.legend()

    # Save the figure instead of displaying it
    plt.savefig(f"training_results_lr_{learning_rate}.png", dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    # define gpu(cuda) if present else use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setting up transformations for data preprocessing and normalizing
    # with mean & standard deviation and transforming to tensors
    transform = transforms.Compose([
        # randomly flip images
        transforms.RandomHorizontalFlip(),
        # Randomly crop the image with padding
        transforms.RandomCrop(32, padding=4),
        # converting images to pytorch tensors
        transforms.ToTensor(),
        # mean and standard deviation unknown so for mean 0.5,0.5,0.5 channels mean is centered
        # towards 0 by subtracting 0.5
        # std of 0.5,0.5,0.5 means that it is divided by 0.5 for each channel to normalize them
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # downloading CIFAR-10 dataset using torchvision.datasets
    train_data = torchvision.datasets.CIFAR10(root='./Data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./Data', train=False, download=True, transform=transform)

    # splitting the dataset into train and test
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

    # learning rate set
    learning_rates = [0.1, 0.001, 0.01, ]
    # loop through the learning rates
    for learning_rate in learning_rates:
        print(f"\nTraining with learning rate: {learning_rate}")
        # instantiating the model
        model = CNNModel().to(device)
        # loss function is the cross entropy
        criterion = nn.CrossEntropyLoss()  # change the name
        # adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # lists to store training accuracy and loss with testing accuracy
        training_loss_history = []
        training_acc_history = []
        testing_acc_history = []
        # specifying num_epochs here
        num_epochs = 20
        # train
        train_model(num_epochs=num_epochs)

        # Call the function to plot and save the training results
        plotting_training_testing_result(num_epochs, training_loss_history, training_acc_history,
                                         testing_acc_history, learning_rate=learning_rate)

        # Save the model parameters
        torch.save(model.state_dict(), f"cnn_model_lr_{learning_rate}.pth")
