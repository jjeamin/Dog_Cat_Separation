import torch
import torch.nn as nn
from torchsummary import summary
from src.converter import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature = nn.Sequential(
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

        self.classify = nn.Sequential(
            nn.Linear(16*16*128, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x


def test():
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


def prune_model(model, filters, last_prune=False):
    conv_id = []
    bn_id = []

    for i, m in model.feature.named_children():
        if type(m) == nn.Conv2d:
            conv_id.append(int(i))
        elif type(m) == nn.BatchNorm2d:
            bn_id.append(int(i))

    if len(conv_id) is not len(filters):
        AssertionError("Conv do not match")

    if last_prune:
        for i, (c_id, b_id) in enumerate(zip(conv_id, bn_id)):
            if i == 0:
                new_conv = cvt_first_conv2d(model.feature[c_id], filters[i])
            else:
                new_conv = cvt_middle_conv2d(model.feature[c_id], filters[i-1], filters[i])

            new_bn = cvt_bn2d(model.feature[b_id], filters[i])

            model.feature[c_id] = new_conv
            model.feature[b_id] = new_bn

        model.classify[0] = cvt_linear(model.classify[0], filters[-1])

    else:
        for i, (c_id, b_id) in enumerate(zip(conv_id, bn_id)):
            if i == 0:
                new_conv = cvt_first_conv2d(model.feature[c_id], filters[i])
                new_bn = cvt_bn2d(model.feature[b_id], filters[i])
            elif i == (len(filters) - 1):
                new_conv = cvt_last_conv2d(model.feature[c_id], filters[i - 1])
                new_bn = cvt_last_bn2d(model.feature[b_id])
            else:
                new_conv = cvt_middle_conv2d(model.feature[c_id], filters[i - 1], filters[i])
                new_bn = cvt_bn2d(model.feature[b_id], filters[i])

            model.feature[c_id] = new_conv
            model.feature[b_id] = new_bn

    print(model)

    return model
