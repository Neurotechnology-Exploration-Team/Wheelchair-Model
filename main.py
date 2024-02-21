import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from EEGDataSet import EEGDataSet
from EEGModel import EEGModel
import glob

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
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


def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            labels = labels.to(dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        avg_valid_loss, valid_accuracy = validate(model, valid_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%')

def main():
    file_paths = glob.glob('cata/*.csv')
    dataset = EEGDataSet(file_paths,exclude_labels=[])

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    model = EEGModel(input_size=8, hidden_size=128, num_layers=2, num_classes=len(set(dataset.labels)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

    train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50)

if __name__ == "__main__":
    main()



# What if I create data sets for every single test