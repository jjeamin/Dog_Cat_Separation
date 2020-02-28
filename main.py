import os
import numpy as np
import matplotlib.pyplot as plt
from src.search import Search
from src.utils import get_cat_dog_path


model_path = './models/model.pth'
cat_paths, dog_paths = get_cat_dog_path('./datasets/test/')

# backprop & get gradient
cat_search = Search(model_path,
                    cat_paths,
                    0)
# backprop & get gradient
dog_search = Search(model_path,
                    dog_paths,
                    1)
cat_total_diffs = cat_search.get_diffs()
dog_total_diffs = dog_search.get_diffs()


for i, j in zip(dog_total_diffs, cat_total_diffs):
    plt.plot(i)
    plt.plot(j)
    plt.show()

