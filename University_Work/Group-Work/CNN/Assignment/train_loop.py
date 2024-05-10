import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import torch
def train(device, model, dataloader, optimizer, criterion, binary=True, unsqueeze=False):
    model.train() 
    
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # Iterate over the training dataset
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if unsqueeze:
            labels=labels.unsqueeze(1).float()
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute the loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        
        running_loss += loss.item()
         # Compute the predicted labels
        if binary:
            predicted = torch.round(outputs)
        else:
            outputs = torch.softmax(outputs, 1)
            _, predicted = torch.max(outputs, 1)
        # Count correct and total predictions
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        

    # Calculate the average loss for the epoch
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = (correct_predictions / total_predictions) * 100
    return epoch_loss, epoch_accuracy

def validate(device, model, dataloader, criterion, binary=True, unsqueeze=False):
    running_loss_val = 0.0
    correct_predictions_val = 0
    total_predictions_val = 0
     # Set the model to validation mode
    model.eval() 
    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over validation set
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if unsqueeze:
                labels=labels.unsqueeze(1).float()
            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            running_loss_val += loss.item()

            # Compute the predicted labels
            if binary:
                predicted = torch.round(outputs)
            else:
                outputs = torch.softmax(outputs, 1)
                _, predicted = torch.max(outputs, 1)
            # Count correct and total predictions
            total_predictions_val += labels.size(0)
            correct_predictions_val += (predicted == labels).sum().item()
                               

    # Calculate the average loss and accuracy
    epoch_loss_val = running_loss_val / len(dataloader)
    epoch_accuracy_val = (correct_predictions_val / total_predictions_val) * 100
    return epoch_loss_val, epoch_accuracy_val

def train_for_epochs(device, epchs, model, trainloader, valloader, optimizer, criterion, binary=True, unsqueeze=False):
    # dictionary for saving history during training
    history = {
        "train": {
            "loss": [],
            "accuracy": []
        },
        "val": {
            "loss": [],
            "accuracy": []
        }
    }
    loop = trange(epchs)
    # Training loop
    for epoch in loop:

        epoch_loss_train, epoch_accuracy_train = train(device, model, trainloader, optimizer, criterion, binary=binary, unsqueeze=unsqueeze)   

        history["train"]["loss"].append(epoch_loss_train)
        history["train"]["accuracy"].append(epoch_accuracy_train)

        epoch_loss_val, epoch_accuracy_val = validate(device, model, valloader, criterion, binary=binary, unsqueeze=unsqueeze)   

        history["val"]["loss"].append(epoch_loss_val)
        history["val"]["accuracy"].append(epoch_accuracy_val)

        loop.set_description(f"Train Loss: {epoch_loss_train:.2f}, Train Acc: {epoch_accuracy_train:.2f}, Val Loss: {epoch_loss_val:.2f}, Val Acc: {epoch_accuracy_val:.2f}")
    return history

def plot_history(loss_train_hist, acc_train_hist, loss_val_hist, acc_val_hist):
    x_arr = np.arange(len(loss_train_hist)) + 1

    with plt.ioff():
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x_arr, loss_train_hist, '-o', label='Train loss')
        ax.plot(x_arr, loss_val_hist, '--<', label='Validation loss')
        ax.set_xlabel('Epoch', size=15)
        ax.set_ylabel('Loss', size=15)
        ax.legend(fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x_arr, acc_train_hist, '-o', label='Train acc.')
        ax.plot(x_arr, acc_val_hist, '--<', label='Validation acc.')
        ax.legend(fontsize=15)
        ax.set_xlabel('Epoch', size=15)
        ax.set_ylabel('Accuracy', size=15)
    return None

