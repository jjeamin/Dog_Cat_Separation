from src.search import Search

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
