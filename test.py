import torch
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

batch_size = 32

load_path = './models/model.pth'

transformer = transforms.Compose([transforms.Resize((128, 128)),
                                  transforms.ToTensor()])

test_dateset = DogCat(dataType='test', transformer=transformer)
test_loader = DataLoader(test_dateset, batch_size=batch_size, shuffle=True)

# model
model = Model().to(device)

model.load_state_dict(torch.load(load_path))

model.eval()

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

    print(f"[TEST Acc / {test_acc}] [TEST Loss / {test_loss}]")

print(dog_cnt)
print(cat_cnt)
