import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.loader import Dog, Cat
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

lr = 0.001
n_epoch = 30
batch_size = 64

save_path = './models/model2.pth'

transformer = transforms.Compose([transforms.Resize((128, 128)),
                                  transforms.ToTensor()])

dog_dataset = Dog(dataType='train', transformer=transformer)
dog_loader = DataLoader(dog_dataset, batch_size=batch_size, shuffle=True)

cat_dataset = Cat(dataType='train', transformer=transformer)
cat_loader = DataLoader(cat_dataset, batch_size=batch_size, shuffle=True)

for dog_img, cat_img in zip(dog_loader, cat_loader):
    plt.imshow

    break

