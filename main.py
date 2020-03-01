from tqdm import tqdm
import pickle
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.loader import DogCat
from src.search import get_filter_idx
from src.utils import get_cat_dog_path, save_pkl, load_pkl
from src.model import prune_model, load_model

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')


def train(model, batch_size, lr=0.001):
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

        print(f"[TRAIN Acc / {train_acc}] [TRAIN Loss / {train_loss}]")

    print(f"[Predict Dog : {dog_cnt}] [Predict Cat : {cat_cnt}]")

    return model


def test(model, batch_size):
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

        print(f"[TEST Acc / {test_acc}] [TEST Loss / {test_loss}]")

    print(f"[Predict Dog : {dog_cnt}] [Predict Cat : {cat_cnt}]")


def gen_filter_idx(model, dataset_path):
    cat_paths, dog_paths = get_cat_dog_path(dataset_path)
    cat_filter, dog_filter = get_filter_idx(model,
                                            cat_paths,
                                            dog_paths)

    for c, d in zip(cat_filter, dog_filter):
        print(f"[Num of Cat Filter : {len(c)}] [Num of Dog Filter : {len(d)}]")

    save_pkl(cat_filter, "./pkl/cat_filter.pkl")
    save_pkl(dog_filter, "./pkl/dog_filter.pkl")


if __name__ == "__main__":
    model_path = './models/model.pth'
    dataset_path = './datasets/test/'

    model = load_model(model_path, mode='eval')

    # gen_filter_idx(model, dataset_path)

    dog_filter = load_pkl("./pkl/dog_filter.pkl")
    cat_filter = load_pkl("./pkl/cat_filter.pkl")

    model = prune_model(model, cat_filter, last_prune=False)
    model = train(model, 32)
    test(model, 32)
