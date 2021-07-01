from os.path import join
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback
from keras import backend as K
import os
import src.utils.const as const
from src.utils.ClusterScoresCallback import ClusterScoresCallbacks



def clear_session():
    # use this for multiple use of networks
    # for some reason destroys good networks
    K.clear_session()


class AuxModelCheckpoint(Callback):
    """
    handle the issue with multiprocessing and saving model,
     copied from https://github.com/keras-team/keras/issues/11101
    """
    def __init__(self, filepath):
        super(AuxModelCheckpoint, self).__init__()
        self.filepath = filepath

        path, filename = os.path.split(self.filepath)
        new_filename = filename.replace('-temp', '')
        new_filepath = os.path.join(path, new_filename)

        self.new_filepath = new_filepath

    def on_epoch_end(self, epoch, logs={}):
        if os.path.exists(self.filepath):
            os.rename(self.filepath, self.new_filepath)

class ClusterBoard(TensorBoard):
    """
    save intermediate cluster values to tensorboard
    """
    def __init__(self, log_dir, model, batch_size, train_callback : ClusterScoresCallbacks, validation_callback : ClusterScoresCallbacks =None, test_callback : ClusterScoresCallbacks =None, extra_names = [], extra_callbacks = []):  # add other arguments to __init__ if you  need
        super().__init__(log_dir=log_dir, batch_size=batch_size)

        self.training_callback = train_callback
        self.validation_callback = validation_callback
        self.test_classback = test_callback
        self.m_model = model

        self.extra_names = extra_names
        self.extra_callbacks = extra_callbacks

        num_subheads = self.training_callback.subheads_number


    def on_epoch_end(self, epoch, logs=None):

        """
        log new cluster scores based an cluster scores callbacks
        :param epoch:
        :param logs:
        :return:
        """

        logs.update({'batch-size': self.batch_size})
        config_opt = self.m_model.optimizer.get_config()
        logs.update({'learn': config_opt['learning_rate']})

        for callback in [self.training_callback, self.validation_callback] + self.extra_callbacks:
            logs.update(callback.scores)

        super().on_epoch_end(epoch, logs)


def callbacks(config, model, mapping_gen, val_gen, extra_names, extra_gens, frozen_backbone=False):
    """
    :param config: config map for configuration details,
    used entries: experiment_id, sub_experiment_name, epochs, batch_size, score_name, with_early_stopping, save_intermediate_weights
    :return:
    """


    log_dir = config.log_dir

    batch_size = config.frozen_batch_size if frozen_backbone else config.batch_size  # use different batch size during frozen backbone

    # use normal tensorboard if clusterer is not used
    if config.cluster_prediction_step < 0:
        callbacks_array = [ TensorBoard(log_dir=log_dir, batch_size=batch_size),]
    else:
        # define possible cluster scores

        mapping_callback_on_training = ClusterScoresCallbacks(const.TRAIN,mapping_gen, model, config)

        extra_callbacks = []
        for data_split_name, gen in zip(extra_names,extra_gens):
            extra_callbacks.append(ClusterScoresCallbacks(data_split_name, gen, model, config))

        if config.use_only_train_cl_acc:
            callbacks_array = [mapping_callback_on_training, *extra_callbacks,  ClusterBoard(log_dir=log_dir, model=model, batch_size=batch_size, train_callback=mapping_callback_on_training, extra_names = extra_names, extra_callbacks=extra_callbacks), ]
        else:
            validation_callback = ClusterScoresCallbacks(const.VAL,val_gen, model, config, training_callback=mapping_callback_on_training)
            callbacks_array = [mapping_callback_on_training, validation_callback, *extra_callbacks,
                                ClusterBoard(log_dir=log_dir, model=model,  batch_size=batch_size, train_callback=mapping_callback_on_training, validation_callback=validation_callback, extra_names = extra_names, extra_callbacks=extra_callbacks), ]

    callbacks_array += [
        ModelCheckpoint(join(log_dir,"weights-temp.h5"), monitor=config.score_name,
                        save_best_only=True,
                        save_weights_only=True, verbose=1),
        ModelCheckpoint(join(log_dir, "model-temp.h5"), monitor=config.score_name,
                        save_best_only=True,
                        save_weights_only=False, verbose=1),
        AuxModelCheckpoint(join(log_dir,"weights-temp.h5")),
        AuxModelCheckpoint(join(log_dir,"model-temp.h5")),
        ReduceLROnPlateau(factor=0.1, monitor=config.score_name, patience=config.loss_reduction_patience),
    ]

    if config.with_early_stopping:
        callbacks_array += [
            EarlyStopping(monitor=config.score_name, patience=config.epoch // 3, min_delta=0.00001, verbose=1),
        ]


    return callbacks_array
