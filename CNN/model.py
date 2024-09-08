import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.output = nn.Sequential(
            # Input: (227×227×1)
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11, 11), stride=4), # (55×55×96)
            nn.ReLU(), # (55×55×96)
            nn.MaxPool2d(kernel_size=(3, 3), stride=2), # (27×27×96)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2), # (27×27×256)
            nn.ReLU(), # (27×27×256)
            nn.MaxPool2d(kernel_size=(3, 3), stride=2), # (13×13×256)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1), # (13×13×384)
            nn.ReLU(), # (13×13×384)
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1), # (13×13×384)
            nn.ReLU(), # (13×13×384)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1), # (13×13×256)
            nn.ReLU(), # (13×13×256)
            nn.MaxPool2d(kernel_size=(3, 3), stride=2), # (6×6×256)
            nn.Flatten(), # (9216)
            nn.Linear(6*6*256, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        return self.output(x)
