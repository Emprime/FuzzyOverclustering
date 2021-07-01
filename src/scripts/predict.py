from os.path import join
from src.archs.general import get_model_with_architecture
from src.data_loaders.general import get_data_generators_for_key
from src.utils import const, util
from src.scripts import keras_callbacks
from src.utils.ClusterScoresCallback import ClusterScoresCallbacks
from src.utils.const import VAL, TRAIN


def predict(config, predict_only):

    print("predict best results")

    data_directory = config.dataset_root
    # batch_size = config.batch_size
    workers = config.workers
    pred_include_unlabeled = config.pred_include_unlabeled
    pred_include_test = config.pred_include_test


    # reset tensorflow and load model
    keras_callbacks.clear_session()

    model_function = get_model_with_architecture(config.model)

    restart = config.restart if predict_only else False # in general do not restart, only if you are predict only

    model, alternate_models = model_function(config, frozen_backbone=False, before_frozen=config.before_frozen,
                                                                 restart= restart,
                                                                 continueWeights=False if restart else True) # in general continue weights (load best result) but in case of restart, really reload


    # calculate input sizes
    in_dimensions = model.layers[0].input_shape
    img_width, img_height = in_dimensions[1], in_dimensions[2]
    target_size = (img_width, img_height)

    # define data directories
    train_dir = join(data_directory, const.TRAIN)
    val_dir = join(data_directory, const.VAL)
    test_dir = join(data_directory, const.TEST)
    unlabeled_dir = config.unlabeled_data_dir

    extra_dirs = []
    extra_names = []

    if pred_include_test:
        extra_names.append("test")
        extra_dirs.append(test_dir)
    if pred_include_unlabeled:
        extra_names.append("unlabeled")
        extra_dirs.append(unlabeled_dir)

    train_gen, mapping_gen, val_gen, extra_gens = get_data_generators_for_key(train_dir, val_dir,
                                                                  target_size, False, config,extra_names=extra_names, extra_dirs = extra_dirs,adaptive_bs=True)


    # combine all used generators
    names = [TRAIN, VAL] + extra_names
    generators = [mapping_gen, val_gen] + extra_gens

    for data_split_name, generator in zip(names, generators):

        cl = ClusterScoresCallbacks(data_split_name, generator, model, config,  use_graph=False)

        preds = cl.on_epoch_end(-1, force_logging=True)

        # TODO insert own metrics here




