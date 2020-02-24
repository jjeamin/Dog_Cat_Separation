import os
from torch.utils.data import Dataset
from PIL import Image


class DogCat(Dataset):
    def __init__(self,
                 dataType='train',
                 transformer=None):

        if dataType == 'train':
            img_path = './datasets/train/'
        else:
            img_path = './datasets/test/'

        self.transformer = transformer
        self.imgs = []
        self.labels = []

        for img_name in os.listdir(img_path):
            self.imgs.append(os.path.join(img_path, img_name))
            if img_name.split('.')[0] == 'cat':
                self.labels.append(0)
            else:
                self.labels.append(1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        label = self.labels[idx]

        if self.transformer is not None:
            img = self.transformer(img)

        return img, label

"""
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transformer = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor()])

    dataset = DogCat(dataType='train', transformer=transformer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for i, (img, label) in enumerate(dataloader):
        print(img.shape, label)
        break
"""
