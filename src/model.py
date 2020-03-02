import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary
from src.converter import *
from torch.utils.data import DataLoader
from datasets.loader import DogCat
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score


if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')


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


def prune_model(model, filters, last_prune=False, cls=None):
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

    if cls is not None:
        model.classify[-1] = cvt_binary_linear(model.classify[-1], cls)

    return model


def binary_train(model, batch_size, cls, logger, lr=0.001):
    model.train()

    transformer = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(size=(128, 128), padding=4),
                                      transforms.ToTensor()])

    train_dataset = DogCat(dataType='train', transformer=transformer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    train_iter = len(train_loader)

    train_loss = 0

    train_f1_score = 0
    train_precision = 0
    train_recall = 0

    for i, (images, labels) in tqdm(enumerate(train_loader), total=train_iter):
        images, labels = images.to(device), labels.to(device)
        # lab : [[1][0][1]]
        # cat : [[0][1][0]]
        # dog : [[1][0][1]]
        labels = labels.unsqueeze(dim=1)

        new_labels = torch.zeros_like(labels)
        new_labels[labels == cls] = 1

        optimizer.zero_grad()
        # forward
        pred = model(images)
        # acc
        p = pred.squeeze().cpu() > 0.5
        t = new_labels.squeeze().cpu()

        train_f1_score += f1_score(p, t, average="binary")
        train_precision += precision_score(p, t, average="binary")
        train_recall += recall_score(p, t, average="binary")

        # loss
        loss = criterion(pred, new_labels.float())
        train_loss += loss.item()
        # backward
        loss.backward(retain_graph=True)
        # weight update
        optimizer.step()

    f1 = train_f1_score / train_iter
    precision = train_precision / train_iter
    recall = train_recall / train_iter

    train_loss = train_loss / train_iter

    logger.info(f"TRAIN [F1_score / {f1}] , [Precision / {precision}] : [recall / {recall}] : [Loss /  {train_loss}]")

    return model


def binary_test(model, batch_size, cls, logger,):
    model.eval()

    transformer = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor()])

    test_dateset = DogCat(dataType='test', transformer=transformer)
    test_loader = DataLoader(test_dateset, batch_size=batch_size, shuffle=True)

    # cost
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    test_iter = len(test_loader)

    test_f1_score = 0
    test_precision = 0
    test_recall = 0
    test_loss = 0

    for i, (images, labels) in tqdm(enumerate(test_loader), total=test_iter):
        images, labels = images.to(device), labels.to(device)

        labels = labels.unsqueeze(dim=1)

        new_labels = torch.zeros_like(labels)
        new_labels[labels == cls] = 1
        # forward
        pred = model(images)
        # acc
        p = pred.squeeze().cpu() > 0.5
        t = new_labels.squeeze().cpu()

        test_f1_score += f1_score(p, t, average="binary")
        test_precision += precision_score(p, t, average="binary")
        test_recall += recall_score(p, t, average="binary")
        # loss
        loss = criterion(pred, new_labels.float())
        test_loss += loss.item()

    f1 = test_f1_score / test_iter
    precision = test_precision / test_iter
    recall = test_recall / test_iter

    test_loss = test_loss / test_iter

    logger.info(f"TEST [F1_score / {f1}] , [Precision / {precision}] : [recall / {recall}] : [Loss /  {test_loss}]")


def train(model, batch_size, logger, lr=0.001):
    model.train()
    transformer = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(size=(128, 128), padding=4),
                                      transforms.ToTensor()])

    train_dataset = DogCat(dataType='train', transformer=transformer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    train_iter = len(train_loader)

    train_loss = 0
    n_train_correct = 0

    dog_cnt = 0
    cat_cnt = 0

    for i, (images, labels) in tqdm(enumerate(train_loader), total=train_iter):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # forward
        pred = model(images)
        # acc
        _, predicted = torch.max(pred, 1)

        for l, p in zip(labels, predicted):
            if l.item() == 0:
                if p.item() == l.item():
                    cat_cnt += 1
            else:
                if p.item() == l.item():
                    dog_cnt += 1

        n_train_correct += (predicted == labels).sum().item()
        # loss
        loss = criterion(pred, labels)
        train_loss += loss.item()
        # backward
        loss.backward(retain_graph=True)
        # weight update
        optimizer.step()

        train_acc = n_train_correct / (train_iter * batch_size)
        train_loss = train_loss / train_iter

    logger.info(f"[TRAIN Acc / {train_acc}] [TRAIN Loss / {train_loss}]")
    logger.info(f"[Predict Dog : {dog_cnt}] [Predict Cat : {cat_cnt}]")

    return model


def test(model, batch_size, logger,):
    model.eval()
    transformer = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor()])

    test_dateset = DogCat(dataType='test', transformer=transformer)
    test_loader = DataLoader(test_dateset, batch_size=batch_size, shuffle=True)

    # cost
    criterion = torch.nn.CrossEntropyLoss().to(device)

    test_iter = len(test_loader)
    test_loss = 0
    n_test_correct = 0

    dog_cnt = 0
    cat_cnt = 0

    for i, (images, labels) in tqdm(enumerate(test_loader), total=test_iter):
        images, labels = images.to(device), labels.to(device)
        # forward
        pred = model(images)
        # acc
        _, predicted = torch.max(pred, 1)

        for l, p in zip(labels, predicted):
            if l.item() == 0:
                if p.item() == l.item():
                    cat_cnt += 1
            else:
                if p.item() == l.item():
                    dog_cnt += 1

        n_test_correct += (predicted == labels).sum().item()
        # loss
        loss = criterion(pred, labels)
        test_loss += loss.item()

        test_acc = n_test_correct / (test_iter * batch_size)
        test_loss = test_loss / test_iter

    logger.info(f"[TEST Acc / {test_acc}] [TEST Loss / {test_loss}]")
    logger.info(f"[Predict Dog : {dog_cnt}] [Predict Cat : {cat_cnt}]")
