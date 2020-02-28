import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.loader import DogCat
from src.model import Model
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

lr = 0.001
n_epoch = 30
batch_size = 64

save_path = './models/model.pth'

transformer = transforms.Compose([transforms.Resize((128, 128)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomCrop(size=(128, 128),padding=4),
                                  transforms.ToTensor()])

train_dataset = DogCat(dataType='train', transformer=transformer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = DogCat(dataType='test', transformer=transformer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# model
model = Model().to(device)
model.load_state_dict(torch.load(save_path))

# cost
criterion = torch.nn.CrossEntropyLoss().to(device)

# optimizer/scheduler
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

train_iter = len(train_loader)
test_iter = len(test_loader)
best_acc = 0

for e in range(n_epoch):
    train_loss = 0
    test_loss = 0
    n_train_correct = 0
    n_test_correct = 0

    for i, (images, labels) in tqdm(enumerate(train_loader), total=train_iter):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # forward
        pred = model(images)
        # acc
        _, predicted = torch.max(pred, 1)
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

    print(f"[TRAIN Acc / {train_acc}] [TRAIN Loss / {train_loss}]")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # forward
            pred = model(images)
            # acc
            _, predicted = torch.max(pred, 1)
            n_test_correct += (predicted == labels).sum().item()
            # loss
            loss = criterion(pred, labels)
            test_loss += loss.item()

        test_acc = n_test_correct / (test_iter * batch_size)
        test_loss = test_loss / test_iter

        print(f"[TEST Acc / {test_acc}] [TEST Loss / {test_loss}]")

        if test_acc > best_acc:
            print("model saved")
            torch.save(model.state_dict(), save_path)
            best_acc = test_acc
