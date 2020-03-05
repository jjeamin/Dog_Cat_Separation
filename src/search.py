import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn import functional as F
from src.utils import get_cat_dog_path, pil_to_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Search(object):
    def __init__(self,
                 model,
                 image_paths,
                 label):
        self.model = model
        self.image_paths = image_paths
        self.label = label
        self.total_diffs = 0
        self.using = 0

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
        # print("[ Grad INFO ]")
        grads = []

        for m in self.model.modules():
            if type(m) == nn.Conv2d:
                # print(f"Grad Shape : {m.weight.grad.shape}")
                grads.append(m.weight.grad.cpu().detach().numpy())

        # print(f"Num Conv2d Layer : {len(grads)}")

        return grads

    def backprop(self, image_path, inverse=False):
        self.model.zero_grad()
        img = pil_to_tensor(Image.open(image_path))
        # forward
        output = self.model(img)
        # acc
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        if pred is not self.label:
            return None

        """
        # 0, 1
        if inverse:
            pred = 1 if pred == 0 else 0

        one_hot_output = torch.zeros(1, h_x.size()[0]).to(device)
        one_hot_output[0][pred] = 1

        output.backward(gradient=one_hot_output)
        """

        # prob, prob
        output.backward(gradient=h_x.flip(0).unsqueeze(0)) if inverse else output.backward(gradient=h_x.unsqueeze(0))

        grads = self.get_conv_grad()

        return grads

    def set_diffs(self):
        for image_path in self.image_paths:
            diffs = []

            t_grad = self.backprop(image_path)
            f_grad = self.backprop(image_path, inverse=True)

            if t_grad is None:
                continue

            self.using += 1

            for (t, f) in zip(t_grad, f_grad):
                diffs.append(self.get_diff(t, f))

            self.total_diffs += np.array(diffs)

    @staticmethod
    def get_diff(t, f):
        t = abs(t)
        f = abs(f)

        sum_t = t.reshape(t.shape[0], -1).sum(1)
        sum_f = f.reshape(f.shape[0], -1).sum(1)

        return (sum_f - sum_t) / (sum_t + 1e-5)

    def get_diffs(self):
        self.set_diffs()
        print(self.using)
        return self.total_diffs


def get_filter_idx(model, cat_paths, dog_paths, show=True):
    # backprop & get gradient
    cat_search = Search(model,
                        cat_paths,
                        0)

    # backprop & get gradient
    dog_search = Search(model,
                        dog_paths,
                        1)

    cat_total_diffs = cat_search.get_diffs()
    dog_total_diffs = dog_search.get_diffs()

    cat_filter = [[] for _ in range(len(cat_total_diffs))]
    dog_filter = [[] for _ in range(len(dog_total_diffs))]

    for i, (dog, cat) in enumerate(zip(dog_total_diffs, cat_total_diffs)):
        for j, (d, c) in enumerate(zip(dog, cat)):
            if i < 2:
                cat_filter[i].append(j)
                dog_filter[i].append(j)

            else:
                if d > 0 and c > 0:
                     cat_filter[i].append(j)
                     dog_filter[i].append(j)
                elif d > 0:
                     cat_filter[i].append(j)
                elif c > 0:
                     dog_filter[i].append(j)

        if show:
            plt.plot(dog, label="Dog")
            plt.plot(cat, label="Cat")
            plt.legend()
            plt.show()

    return cat_filter, dog_filter


def gen_filter_idx(model, dataset_path, show=True):
    cat_paths, dog_paths = get_cat_dog_path(dataset_path)
    cat_filter, dog_filter = get_filter_idx(model,
                                            cat_paths,
                                            dog_paths,
                                            show=show)

    for c, d in zip(cat_filter, dog_filter):
        print(f"[Num of Cat Filter : {len(c)}] [Num of Dog Filter : {len(d)}]")

    return cat_filter, dog_filter
