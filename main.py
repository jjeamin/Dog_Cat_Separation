import copy
from src.utils import load_pkl ,get_logger
from src.prune import *
from src.vgg import load_model
from src.search import gen_filter_idx

if __name__ == "__main__":
    model_path = './models/vgg11.pth'
    dataset_path = './datasets/test/'
    logger = get_logger('./log3.log')

    model = load_model(model_path, version="VGG11", mode='eval')

    cls = 1

    for _ in range(0, 10):
        cat_filter, dog_filter = gen_filter_idx(model, dataset_path)

        model = prune(model,
                      cat_filter,
                      last_prune=False)

        for _ in range(0, 10):
            model = train(model, batch_size=32, logger=logger)
            test(model, batch_size=32, logger=logger)



    # save_pkl(cat_filter, "./pkl/cat_filter.pkl")
    # save_pkl(dog_filter, "./pkl/dog_filter.pkl")
    #
    # dog_filter = load_pkl("./pkl/dog_filter.pkl")
    # cat_filter = load_pkl("./pkl/cat_filter.pkl")

    # Cat : 0 / Dog : 1
    # cls = [1, 0]
    # filters = [dog_filter, cat_filter]
    # cls_name = ["dog", "cat"]
    #
    # for i, c in enumerate(cls):
    #     for j, f in enumerate(filters):
    #         logger.info(f"[class : {cls_name[i]}] / [filter name : {cls_name[j]}]")
    #
    #         logger.info("[Categorical]")
    #         model = load_model(model_path, mode='eval')
    #         model = prune_model(model,
    #                             f,
    #                             last_prune=False)
    #
    #         for _ in range(0, 10):
    #             model = train(model, batch_size=32, logger=logger)
    #             test(model, batch_size=32, logger=logger)
    #
    #         logger.info("[Binary]")
    #         # model = load_model(model_path, mode='eval')
    #         model = load_model(model_path, version="VGG11", mode='eval')
    #
    #         model = prune_model(model,
    #                             f,
    #                             last_prune=False,
    #                             cls=c)
    #
    #         model = binary_train(model, batch_size=32, cls=c, logger=logger)
    #         binary_test(model, batch_size=32, cls=c, logger=logger)
