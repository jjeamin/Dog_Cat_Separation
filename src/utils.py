import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image


def get_cat_dog_path(root_path):
    cat_paths = []
    dog_paths = []

    for name in os.listdir(root_path):
        if name.split(".")[0] == 'cat':
            cat_paths.append(os.path.join(root_path, name))
        else:
            dog_paths.append(os.path.join(root_path, name))

    return cat_paths, dog_paths


def get_path(root_path):
    paths = [os.path.join(root_path, name) for name in os.listdir(root_path)]

    return paths


def pil_imshow(img):
    plt.imshow(img)
    plt.show(block=True)


def pil_to_tensor(img, size=(128, 128), device='cuda'):
    transformer = transforms.Compose([transforms.Resize(size),
                                      transforms.ToTensor()])
    tensor_img = transformer(img)
    tensor_img = tensor_img.unsqueeze(dim=0).to(device)

    return tensor_img


def gen_mix_img(root_path, mix_path):
    cat_paths, dog_paths = get_cat_dog_path(root_path)

    if os.path.exists(mix_path):
        pass
    else:
        os.mkdir(mix_path)

    cnt = 0

    for cat_path, dog_path in zip(cat_paths, dog_paths):
        cat_img = Image.open(cat_path)
        dog_img = Image.open(dog_path)

        size = (128, 128)

        resize_cat_img = cat_img.resize(size)
        crop_cat_img = resize_cat_img.crop((0, 0, size[0] / 2, size[1]))

        resize_dog_img = dog_img.resize(size)
        crop_dog_img = resize_dog_img.crop((0, 0, size[0] / 2, size[1]))

        mix_img = Image.fromarray(np.hstack((crop_cat_img, crop_dog_img)))
        mix_img.save(os.path.join(mix_path, f'img_{cnt}.jpg'))

        cnt += 1


def show_feature(feature):
    f_len = len(feature)

    size = int(np.sqrt(f_len))

    fig, axs = plt.subplots(size, size, figsize=(10, 10))
    cnt = 0

    for i in range(size):
        for j in range(size):
            axs[i, j].imshow(feature[cnt])
            axs[i, j].axis('off')
            cnt += 1

    plt.show()


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)  # 단 한줄씩 읽어옴

    return data
