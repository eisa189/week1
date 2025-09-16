
import torch

import warnings
warnings.filterwarnings('ignore')
def train(train_loader, optimizer, model, criterion, device):
    model.train()
    total_samples = 0
    total_loss = 0
    correct= 0
    for batch_idx, batch_data in enumerate(train_loader):
            
        images = batch_data['image'].to(device)
        feature_text = batch_data['feature_text'].to(device)
        labels = batch_data['label'].to(device)
        
        optimizer.zero_grad()
        
        img_output = model(images, feature_text)
        
        loss = criterion(img_output, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(img_output, 1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct / total_samples

    return epoch_loss, epoch_accuracy
    
def validation(valid_loader, model, criterion, device):
    model.eval()
    valid_correct = 0
    valid_total_samples = 0
    valid_loss = 0.0
    with torch.no_grad():
        for valid_batch_data in valid_loader:
            valid_images = valid_batch_data['image'].to(device)
            valid_feature_text = valid_batch_data['feature_text'].to(device)
            valid_labels = valid_batch_data['label'].to(device)
            
            valid_output = model(valid_images, valid_feature_text)
            valid_loss += criterion(valid_output, valid_labels).item()
            
            _, valid_predicted = torch.max(valid_output, 1)
            valid_correct += (valid_predicted == valid_labels).sum().item()
            valid_total_samples += valid_labels.size(0)
            
    valid_epoch_loss = valid_loss / len(valid_loader)
    valid_epoch_accuracy = valid_correct / valid_total_samples

    return valid_epoch_loss, valid_epoch_accuracy

def test(test_loader, model, device):
    ctest_labels_lst=[]
    ctest_predicted_lst=[]

    model.eval()
    with torch.no_grad():
        for ctest_batch_data in test_loader:
            ctest_images = ctest_batch_data['image'].to(device)
            ctest_labels = ctest_batch_data['label'].to(device)
            ctest_feature_text = ctest_batch_data['feature_text'].to(device)
            
            ctest_output = model(ctest_images, ctest_feature_text)
            
            _, ctest_predicted = torch.max(ctest_output, 1)
            ctest_labels_lst.extend(ctest_labels.cpu().numpy()) 
            ctest_predicted_lst.extend(ctest_output.cpu().numpy())
    return ctest_labels_lst, ctest_predicted_lst
        