import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from EEGDataSet import EEGDataSet
from EEGModel import EEGModel
import glob

def validate(model, dataloader, criterion,device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.to(dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy



def train(model, train_loader, valid_loader, criterion, optimizer,device, num_epochs=10):
    total_batches = len(train_loader)
    print(f"Starting to train... Total batches: {total_batches}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.to(dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch {batch_idx}/{total_batches}, Current Batch Loss: {loss.item():.4f}')
                
        avg_train_loss = total_loss / total_batches
        avg_valid_loss, valid_accuracy = validate(model, valid_loader, criterion,device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%')


def main():
    print("Initilizing everything")
    file_paths = glob.glob('cata/*.csv')
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available in PyTorch!")
    else:
        print("CUDA (GPU support) is NOT available in PyTorch!")


    print("Creating Data Sets")
    dataset = EEGDataSet(file_paths,exclude_labels=[])

    print("Finished Data Sets")

    print("Creating Data Loaders")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers = 10)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers = 10)
    print("Finished Data Loaders")

    print("Starting model creation")
    model = EEGModel(input_size=5, hidden_size=128, num_layers=2, num_classes=len(set(dataset.labels)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    print("Finished model creation")

    train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=5,device = device)

if __name__ == "__main__":
    main()

