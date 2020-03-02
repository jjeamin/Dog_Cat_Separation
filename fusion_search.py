import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from PIL import Image
from src.model import load_model
from src.utils import get_path, pil_to_tensor, show_feature
from random import randint


class Fusion_Search(object):
    def __init__(self,
                 model_path,
                 image_paths,
                 mode='eval'):
        self.model = load_model(model_path, mode)
        self.image_paths = image_paths
        self.feature = []
        self.r = randint(0, len(image_paths))

    def get_feature_map(self, modele, input, output):
        self.feature.append(output[0].cpu().data.numpy())

    def register(self):
        for m in self.model.modules():
            if type(m) == nn.Conv2d:
                m.register_forward_hook(self.get_feature_map)

    def predict(self):
        self.register()

        img = pil_to_tensor(Image.open(self.image_paths[self.r]))
        print(self.image_paths[self.r])
        # forward
        output = self.model(img)
        # acc
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        print(pred)
        return self.feature


model_path = './models/model.pth'
mix_paths = get_path('./datasets/mix/')

search = Fusion_Search(model_path,
                       mix_paths)

feature = search.predict()

for f in feature:
    dog_cnt = 0
    cat_cnt = 0

    show_feature(f)

    for i in f:
        #i = Image.fromarray(i)
        #i = i.resize((128, 128), Image.BILINEAR)

        width = i.shape[0]
        max_pixel = np.argmax(i)

        if max_pixel % width >= width / 2:
            dog_cnt += 1
        else:
            cat_cnt += 1

    print(f"dog: {dog_cnt}")
    print(f"cat: {cat_cnt}")
