from model import RNN
from data import get_train_data, n_categories, n_letters
import torch
import torch.nn as nn

device = "mps" if torch.backends.mps.is_available() else "cpu"
n_hidden = 128
lr = 0.0001
epochs = 20

model = RNN(input_size=n_letters, hidden_size=n_hidden, output_size=n_categories).to(device)

data = get_train_data()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for i, (X, Y) in enumerate(data):
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()

        hidden = model.initHidden().to(device)
        for i in range(X.size(1)):
            output, hidden = model(X[0][i], hidden)

        J = loss(output, Y)

        
        J.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data)}], Loss: {J.item()}")


