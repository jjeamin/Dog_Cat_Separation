import os
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from src.model import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pil_imshow(img):
    plt.imshow(img)
    plt.show(block=True)


def pil_to_tensor(img, size=(128, 128)):
    transformer = transforms.Compose([transforms.Resize(size),
                                      transforms.ToTensor()])
    tensor_img = transformer(img)
    tensor_img = tensor_img.unsqueeze(dim=0).to(device)

    return tensor_img


class Search(object):
    def __init__(self,
                 model_path,
                 image_path,
                 mode='eval'):
        self.model = load_model(model_path, mode)
        self.image_name = image_path.split('/')[-1].split('.')[0]
        self.image_path = image_path

    def get_conv_weight(self):
        print("[ Weight INFO ]")
        weights = []

        for m in self.model.modules():
            if type(m) == nn.Conv2d:
                print(f"Weight Shape : {m.weight.shape}")
                weights.append(m.weight.cpu().detach().numpy())

        print(f"Num Conv2d Layer : {len(weights)}")

        return weights

    def get_conv_grad(self):
        print("[ Grad INFO ]")
        grads = []

        for m in self.model.modules():
            if type(m) == nn.Conv2d:
                print(f"Grad Shape : {m.weight.grad.shape}")
                grads.append(m.weight.grad.cpu().detach().numpy())

        print(f"Num Conv2d Layer : {len(grads)}")

        return grads

    def backprop(self, inverse=False):
        self.model.zero_grad()
        img = pil_to_tensor(Image.open(self.image_path))
        # forward
        output = self.model(img)
        # acc
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        if inverse:
            pred = 1 if pred == 0 else 0

        one_hot_output = torch.zeros(1, h_x.size()[0]).to(device)
        one_hot_output[0][pred] = 1

        output.backward(gradient=one_hot_output)

        """
        # backprop
        if inverse:
            output.backward(gradient=h_x.flip(0).unsqueeze(0))
        else:
            output.backward(gradient=h_x.unsqueeze(0))
        """

        grads = self.get_conv_grad()

        return grads

    @staticmethod
    def get_diff(t, f):
        t = abs(t)
        f = abs(f)

        sum_t = t.reshape(t.shape[0], -1).sum(1)
        sum_f = f.reshape(f.shape[0], -1).sum(1)

        plt.plot(sum_t)
        plt.plot(sum_f)

        return (sum_f - sum_t) / (sum_t + 1e-5)

    def diff_show(self, t_grad, f_grad, layer=None):
        if layer is not None:
            t = t_grad[layer]
            f = f_grad[layer]

            diff = self.get_diff(t, f)

            plt.plot(diff)
            plt.title(self.image_name.split(".")[0])
            plt.show(block=True)

        else:
            for (t, f) in zip(t_grad, f_grad):
                diff = self.get_diff(t, f)

                plt.plot(diff)
                plt.title(self.image_name.split(".")[0])
                plt.show(block=True)


if __name__ == "__main__":
    MODEL_PATH = './models/model.pth'
    IMAGE_ROOT_PATH = './datasets/test/'
    number = 4211
    IMAGE_NAME = [f'dog.{number}.jpg', f'cat.{number}.jpg']

    for name in IMAGE_NAME:
        # backprop & get gradient
        search = Search(MODEL_PATH,
                        IMAGE_ROOT_PATH,
                        name)

        true_grad = search.backprop()
        false_grad = search.backprop(inverse=True)

        search.diff_show(true_grad, false_grad)
