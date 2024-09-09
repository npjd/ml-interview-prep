from model import RNN
from data import get_train_data, n_categories, n_letters
import torch
import torch.nn as nn

device = "mps" if torch.backends.mps.is_available() else "cpu"
n_hidden = 128
lr = 0.0001
epochs = 1

model = RNN(input_size=n_letters, hidden_size=n_hidden, output_size=n_categories).to(device)

train_data_loader, test_data_loader = get_train_data()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for i, (X, Y) in enumerate(train_data_loader):
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()

        hidden = model.initHidden().to(device)
        for j in range(X.size(1)):
            output, hidden = model(X[0][j], hidden)

        J = loss(output, Y)

        J.backward()
        optimizer.step

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_data_loader)}], Loss: {J.item()}")

with torch.no_grad():
    correct = 0
    for i, (X, Y) in enumerate(test_data_loader):
        X = X.to(device)
        Y = Y.to(device)

        hidden = model.initHidden().to(device)
        for j in range(X.size(1)):
            output, hidden = model(X[0][j], hidden)
        
        print(f"Predicted: {output.argmax().item()} Ground Truth: {Y.argmax().item()}")

        if output.argmax().item() == Y.argmax().item():
            correct += 1
    print(f"Accuracy: {correct/len(test_data_loader)}")