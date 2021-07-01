import argparse
import os
import pickle
from os.path import join
import wandb

from src.scripts.train import train_network
from src.scripts.predict import predict
from src.utils import util, const

from src.utils.util import load_arguments, config_to_str


def main():
    """
    Main launch script to specify parameters and define experiments, convention for parameters, list are plural, parameters are singular
    """
    parser = argparse.ArgumentParser()

    # load arguments based on yaml
    parser = load_arguments(parser, "arguments.yaml")

    arguments = parser.parse_args()

    # special treatment of input for legacy reasons
    if arguments.input == "rgb":
        arguments.sobel_only = False
        arguments.sobel = False
        arguments.sobel_rgb = False
    elif arguments.input == "sobel":
        arguments.sobel_only = True
        arguments.sobel = True
        arguments.sobel_rgb = False
    elif arguments.input == "sobel_rgb":
        arguments.sobel_only = False
        arguments.sobel = True
        arguments.sobel_rgb = True

    # legacy experiment name
    arguments.sub_experiment_name = arguments.IDs[-1]
    arguments.experiment_identifiers = arguments.IDs[:-1]

    arguments.epoch = arguments.frozen_epoch + arguments.normal_epoch
    arguments.use_triplet_loss = True

    root = arguments.datasets_root
    arguments.dataset_root = join(root, arguments.dataset)
    assert os.path.exists(arguments.dataset_root), "The specified data directory is invalid %s" % arguments.dataset_root
    if arguments.unlabeled_data is None or arguments.unlabeled_data == "None":
        arguments.unlabeled_data_dir = None
    else:
        arguments.unlabeled_data_dir = join(root, arguments.unlabeled_data, const.UNLABELED)
        assert os.path.exists(arguments.unlabeled_data_dir), "The specified unlabeled data directory is invalid %s" % arguments.unlabeled_data_dir

    # number gt classes is based on training directory
    class_labels = sorted(os.listdir(join(arguments.dataset_root, const.TRAIN)))
    arguments.gt_k = len(class_labels)
    arguments.class_labels = class_labels


    experiment(arguments)

def experiment(config):
    """
     start selected experiment with given config
    :param config:
    :return:
    """



    # set logging
    config.log_dir, time_name = util.create_log_dir(config.root_log_dir, config.experiment_identifiers, config.sub_experiment_name,
                                         with_timestamp=True)  # create a log dir for this run with time

    tags = config.experiment_identifiers
    config.unique_name = "#".join([tags[-1],config.sub_experiment_name])

    # save config
    with open(join(config.log_dir, 'config.pickle'), 'wb') as f:
        config.graph = None  # rerun bug with graph and pickle
        pickle.dump(config, f)

    # setup
    print("START EXPERIMENT - %s\n" % config.sub_experiment_name)
    print(config_to_str(config))

    # train and predict network
    if config.train:
        train_network(config)

    if config.predict:
        predict(config, predict_only = not config.train)


if __name__ == "__main__":
    main()

