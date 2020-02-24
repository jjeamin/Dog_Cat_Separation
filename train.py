import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from loader import DogCat
from model import Model
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

lr = 0.001
n_epoch = 20
batch_size = 32

save_path = './model.pth'

transformer = transforms.Compose([transforms.Resize((128, 128)),
                                  transforms.ToTensor()])

train_dataset = DogCat(dataType='train', transformer=transformer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# model
model = Model().to(device)

# cost
criterion = torch.nn.CrossEntropyLoss().to(device)

# optimizer/scheduler
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

train_iter = len(train_loader)
best_acc = 0

for e in range(n_epoch):
    train_loss = 0
    n_train_correct = 0

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

    print(f"[TRAIN ACCURACY / {train_acc}] [TRAIN_LOSS / {train_loss}]")

    if train_acc > best_acc:
        print("model saved")
        torch.save(model.state_dict(), save_path)
        best_acc = train_acc
