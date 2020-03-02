from src.utils import load_pkl, get_logger
from src.model import *
from src.search import gen_filter_idx

if __name__ == "__main__":
    model_path = './models/model3.pth'
    dataset_path = './datasets/test/'
    logger = get_logger('./log.log')

    model = load_model(model_path, mode='eval')

    # gen_filter_idx(model, dataset_path)

    dog_filter = load_pkl("./pkl/dog_filter.pkl")
    cat_filter = load_pkl("./pkl/cat_filter.pkl")
    # Cat : 0 / Dog : 1
    cls = [1, 0]
    filters = [dog_filter, cat_filter]
    cls_name = ["dog", "cat"]

    for i, c in enumerate(cls):
        for j, f in enumerate(filters):
            logger.info(f"[class : {cls_name[i]}] / [filter name : {cls_name[j]}]")

            # logger.info("[Categorical]")
            # model = load_model(model_path, mode='eval')
            # model = prune_model(model,
            #                     f,
            #                     last_prune=False)
            #
            # for _ in range(0, 10):
            #     model = train(model, batch_size=32, logger=logger)
            #     test(model, batch_size=32, logger=logger)

            logger.info("[Binary]")
            model = load_model(model_path, mode='eval')
            model = prune_model(model,
                                f,
                                last_prune=False,
                                cls=c)

            for _ in range(0, 10):
                model = binary_train(model, batch_size=32, cls=c, logger=logger)
                binary_test(model, batch_size=32, cls=c, logger=logger)
