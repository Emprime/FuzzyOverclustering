import argparse
from multiprocessing import Process
from datetime import datetime
import os
import time
from shutil import rmtree
import numpy as np
from os.path import join
import pickle

import yaml
from tqdm import tqdm

import src.utils.const as const

colors = 2 * ['gray', 'saddlebrown', 'firebrick', 'red', 'sandybrown', 'gold', 'yellow', 'darkkhaki',
              'lawngreen', 'g', 'darkslategray', 'turquoise', 'skyblue', 'dodgerblue', 'b', 'blueviolet',
              'orchid', 'pink', 'm', 'coral', 'lime']

def check_directory(path, delete_old=True, is_file=False):
    """
    check if directory exists and create otherwise
    :param path: path
    :param delete_old: bool which specifies if an old directory should be deleted
    :param is_file: Cuts the last part of path to create only directory, only works when directory divider is "/"
    :return:
    """
    if is_file:
        path = "/" + join(*(path.split("/")[:-1])) # create new root directory

    # check if exists
    # if true and delete old delete and create
    # if false create

    delete = os.path.exists(path) and delete_old
    create = not os.path.exists(path) or delete

    if delete:
        rmtree(path)
    if create:
        os.makedirs(path, exist_ok=True)
    return create



def get_bs_adaptive(exact_nb,batch_size):
    nb_samples = get_nb(exact_nb, batch_size,silent=True)
    while exact_nb - nb_samples > 0:
        # try to fit batch size
        batch_size = batch_size -1 # this might be imporfmant but we dont want to exclude good options
        nb_samples = get_nb(exact_nb, batch_size,silent=True)
    return nb_samples, batch_size

def get_nb(exact_number, batch_size,silent=False):
    """
    get rounded number for exact number and batchsize
    :param exact_number:
    :param batch_size:
    :return:
    """

    if exact_number == 0 or batch_size == 0:
        return 0
    
    if exact_number < batch_size:
        print("ERROR: Batch size is greater than number samples -> this will cause errors")
        return batch_size

    number = ((int)(exact_number / batch_size)) * batch_size

    if not silent:
        if exact_number - number > 0:
            print(("######################################################\n" +
                   "WARNING: you specified %d items, but with a batch size of %d you will only look at %d items\n"
                   + "######################################################") % (exact_number, batch_size, number))

    return number


def config_to_str(config):
    """
    convert config file to string, copied from IIC project
    :param config:
    :return:
    """
    attrs = vars(config)
    string_val = "Config: -----\n"
    string_val += "\n".join("%s: %s" % item for item in attrs.items())
    string_val += "\n----------"
    return string_val

def create_log_dir(root_log_dir, experiment_identifiers, sub_experiment_name, with_timestamp=True):
    """
    combines experiments names and name to a directory for logging, additional usage of a timestamp is possible, everything related to one experiment should be stored here
    :param experiment_identifiers: arrray with at least one entry
    :param sub_experiment_name:
    :param with_timestamp:
    :return:
    """

    # create directory structure from experiment names
    directory_temp = experiment_identifiers[0]
    for exp_name in experiment_identifiers[1:]:
        directory_temp = join(directory_temp, exp_name)

    # add sub_experiment_name and maybe timestamp
    if with_timestamp:
        time_name = "TIME-%s" % time.strftime("%d-%m-%Y-%H-%M-%S")
        log_dir = join(root_log_dir, directory_temp, sub_experiment_name, time_name)
    else:
        log_dir = join(root_log_dir, directory_temp, sub_experiment_name)

    check_directory(log_dir,delete_old=False)

    if with_timestamp:
        return log_dir, time_name
    else:
        return log_dir

def get_last_log_dir(root_log_dir,experiment_identifiers, sub_experiment_name, ignore_current=False):
    """
    get the last log dir for given experiment names and the config sub_experiment_name
    :param experiment_identifiers:
    :param sub_experiment_name:
    :return:
    """

    # get log dir
    log_dir = create_log_dir(root_log_dir,experiment_identifiers, sub_experiment_name, with_timestamp=False)


    # check for directory with times or without
    folders = os.listdir(log_dir)

    # if there are folders with TIME-* take weights from latest
    weights_directories = []
    for folder in folders:
        if folder.startswith("TIME-"):
            weights_directories.append(folder)

    times = []
    # iterate over all weights and save the times
    for weight_directory in weights_directories:
        time_str = weight_directory.split("TIME-")[1] # time after TIME-
        times.append(time_str)

    # sorted(times)
    times.sort(key=lambda date: datetime.strptime(date, "%d-%m-%Y-%H-%M-%S"))

    # get last time
    if len(times) > 0:
        search_time = times[-2 if ignore_current else -1] #.strftime("%d-%m-%Y-%H-%M-%S") # get last or second last

    # search for the file
    for weight_directory in weights_directories:
        if search_time in weight_directory:
            return join(log_dir,weight_directory)

    # No time found return current directory
    return log_dir

def get_weights_latest(root_log_dir,experiment_identifiers, sub_experiment_name, ignore_current=False):

    return join(get_last_log_dir(root_log_dir,experiment_identifiers, sub_experiment_name, ignore_current=ignore_current), "weights.h5")



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def str2bool(v):
    """
    Cast str to bool, usable in argparse
    :param v:
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_arguments(parser, yaml_file, all_multiple=False):
    """
    add arguments based on yaml file to the argparse
    :param parser:
    :param yaml_file:
    :return:
    """

    with open(yaml_file, 'r') as stream:
        try:
            loaded_yaml = yaml.safe_load(stream)

            # convert arguments to argparse arguments
            general_arguments = loaded_yaml["general"]
            for argument_name in general_arguments:

                info = general_arguments[argument_name]

                kwargs = {}

                if "required" in info:
                    if info["required"]:
                        kwargs = {"required": True}

                if "multiple" in info or all_multiple:
                    if all_multiple or info["multiple"]:
                        kwargs = {"nargs": "+"}

                if "dest" in info:
                    kwargs = {"dest": info["dest"]}


                assert "type" in info,  "Wrong name %s" % argument_name
                assert "description" in info,  "Wrong name %s" % argument_name
                assert "value" in info,  "Wrong name %s" % argument_name

                if info["type"] == "string":
                     kwargs["type"] = str
                elif info["type"] == "int":
                     kwargs["type"] = int
                elif info["type"] == "float":
                     kwargs["type"] = float
                elif info["type"] == "bool":
                     kwargs["type"] = str2bool


                # interpret string none as NONE
                value = info["value"]
                value = value if value != "None" else None

                parser.add_argument("--%s" % argument_name, help=info["description"], default=value,  **kwargs)

            # print(loaded_yaml)

        except yaml.YAMLError as exc:
            print(exc)

    return parser