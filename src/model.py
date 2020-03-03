import torch.nn as nn
from torchsummary import summary


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*16*128, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def summary():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = Model().to(device)

    summary(model, (3, 128, 128))


def load_model(path, mode='eval', device='cuda'):
    model = Model().to(device)
    model.load_state_dict(torch.load(path))

    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()
    else:
        AssertionError("MODE is only train and eval")

    return model



