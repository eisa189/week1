import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataset import split_data, CombinedDataset
from training_fuct import train, validation, test
from utils import transform_train, transform_test, get_performance, seed_everything
from models import get_model_net
import warnings
warnings.filterwarnings('ignore')

# Param
images_path = "./data"
data_path = './data/all_data.csv'
learning_rate = 0.001
seed_everything(1)
batch_size = 32
criterion = nn.CrossEntropyLoss()
num_classes = 6
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data
traindf, validdf, testdf = split_data(data_path)
print(f'Training: {len(traindf)}, Validation: {len(validdf)}, Test: {len(testdf)}')
train_dataset = CombinedDataset(traindf, images_path, transforms=transform_train)
valid_dataset = CombinedDataset(validdf, images_path, transforms=transform_test)
test_dataset = CombinedDataset(testdf, images_path, transforms=transform_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Init model 
model = get_model_net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)


# Training 
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []
best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    epoch_loss, epoch_accuracy = train(train_loader, optimizer, model, criterion, device)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')
    
    valid_epoch_loss, valid_epoch_accuracy = validation(valid_loader, model, criterion, device)
    valid_losses.append(valid_epoch_loss)
    valid_accuracies.append(valid_epoch_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_epoch_loss:.4f}, Validation Accuracy: {valid_epoch_accuracy:.4f}')

    if (valid_epoch_loss < best_val_loss):
            best_val_loss = valid_epoch_loss
            torch.save(model.state_dict(),'combine_att_best_model.pt')
            print("Combine Model saved")

model.load_state_dict(torch.load('combine_att_best_model.pt'))
ctest_labels_lst, ctest_predicted_lst = test(test_loader, model, device)
ctest_labels_np = np.array(ctest_labels_lst)
ctest_predicted_np = np.array(ctest_predicted_lst)
accuracy, precision, recall, f1 = get_performance(ctest_labels_np, np.argmax(ctest_predicted_np, axis=1))

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')