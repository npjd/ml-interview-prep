from model import CNN
from data import load_test_data, load_train_validation_data
import torch.nn as nn 
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

num_classes = 10
num_epochs = 20
batch_size = 64
learning_rate = 0.005

model = CNN(num_classes=num_classes).to(device)
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(f"{name}: {parameter.shape}")

print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

train_dataset, validation_dataset = load_train_validation_data(
    data_dir='data',
    validation_size=0.1,
    random_seed=42,
    batch_size=batch_size,
    shuffle=True,
)

total_step = len(train_dataset)

test_dataset = load_test_data(
    data_dir='data',
    shuffle=True,
    batch_size=batch_size,
)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataset):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        J = loss(outputs, labels)
        
        optimizer.zero_grad()
        J.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {J.item()}")

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in validation_dataset:
            images = images.to(device)
            labels = labels.to(device) 
            outputs = model(images)
            # dim = 1 means we want to get the max value across the columns (max values and idices)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            # item makes it a scalar
            correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total}")
    
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataset:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))   
