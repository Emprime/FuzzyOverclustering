import warnings
from os.path import join
from keras.engine.training_utils import iter_sequence_infinite, is_sequence, should_run_validation
from keras.optimizers import Adam, SGD
from keras.utils import OrderedEnqueuer, GeneratorEnqueuer
from keras.utils.generic_utils import to_list
from src.archs.general import get_model_with_architecture
from src.utils.losses import metric_accuracy, metric_ig_first, metric_ig_second, metric_ce, metric_ce_inv, triplet_loss
from src.data_loaders.general import get_data_generators_for_key
from src.utils import const
from src.scripts import keras_callbacks
import keras.callbacks as cbks
import keras.backend as K


def train_network(config):
    """
    train the model based on the frozen and normal epochs
    :param config:
    :return:
    """

    # setup
    model_function = get_model_with_architecture(config.model)
    before_frozen = config.before_frozen
    fep = config.frozen_epoch
    nep = config.normal_epoch
    ep = config.epoch
    restart = config.restart

    if fep > 0:


        #  train frozen
        print("start frozen training")
        keras_callbacks.clear_session()


        model, alternate_models = model_function(config, frozen_backbone=True, before_frozen=before_frozen, restart=restart, )
        train_with_generator(config, start_epoch=0, end_epoch=fep , model=model,alternate_models=alternate_models, frozen_backbone=True)


        # start rest
        if ep - fep > 0:
            keras_callbacks.clear_session()

            print("start rest of training")
            model, alternate_models = model_function(config, frozen_backbone=False, before_frozen=True, continueWeights=True)
            train_with_generator(config, start_epoch=fep, end_epoch=ep, model=model,  alternate_models=alternate_models, frozen_backbone=False)

    else:
        print("start normal training")
        keras_callbacks.clear_session()

        model, alternate_models = model_function(config, frozen_backbone=False, before_frozen=before_frozen,  restart=restart)

        train_with_generator(config, start_epoch=0, end_epoch=ep, model=model, alternate_models=alternate_models, frozen_backbone=False)


def train_with_generator(config, start_epoch, end_epoch, model, alternate_models, frozen_backbone):


    workers = config.workers * config.num_gpus # use more workers with more gpus
    data_directory = config.dataset_root

    # calculate input sizes
    in_dimensions = model.layers[0].input_shape
    img_width, img_height = in_dimensions[1], in_dimensions[2]
    target_size = (img_width, img_height)

    # define data directories
    train_dir = join(data_directory, const.TRAIN)
    val_dir = join(data_directory, const.VAL)
    test_dir = join(data_directory, const.TEST)

    overcluster_number = config.subheads_number
    gt_number = config.subheads_gt_number
    subheads_number = overcluster_number + gt_number

    adaptive_bs = config.adaptive_bs

    opt = config.optimizer
    lr = config.lr
    frozen_lr = config.frozen_lr

    # parameter compiling

    if frozen_backbone:
        learning_rate_to_use = frozen_lr
    else:
        learning_rate_to_use = lr

    if opt == 'adam':
        opt = Adam(learning_rate_to_use)
        # opt = tf.train.AdamOptimizer(learning_rate_to_use) # early stopping
    elif opt == 'sgd':
        opt = SGD(learning_rate_to_use)
        # opt = tf.train.GradientDescentOptimizer(learning_rate_to_use) # used for eager exectution
    else:
        raise ValueError("Optimizer %s not defined" % opt)

    # loss and metrics

    metrics = [metric_accuracy(config.use_triplet_loss),
               metric_ig_first(config.use_triplet_loss, config.lambda_m),
               metric_ig_second(config.use_triplet_loss, config.lambda_m),
               metric_ce(config.use_triplet_loss, config.lambda_s),
               metric_ce_inv(config.use_triplet_loss, config.lambda_t)]


    pred_include_unlabeled = config.pred_include_unlabeled
    pred_include_test = config.pred_include_test
    unlabeled_dir = config.unlabeled_data_dir

    extra_dirs = []
    extra_names = []

    if pred_include_test:
        extra_names.append("test")
        extra_dirs.append(test_dir)
    if pred_include_unlabeled:
        extra_names.append("unlabeled")
        extra_dirs.append(unlabeled_dir)

    # ACTUAL TRAINING
    # ALTERNATING
    ac = 0
    ao = config.alternate_overcluster_max
    ag = config.alternate_gt_max


    # loss
    normal_loss_function =  triplet_loss(config.lambda_s, 0, config.lambda_m)
    overcluster_loss_function = triplet_loss(0, config.lambda_t, config.lambda_m)


    assert overcluster_number == gt_number

    normal_losses_half = [normal_loss_function for i in range(overcluster_number)]  # same loss for all subheads
    over_losses_half = [overcluster_loss_function for i in range(gt_number)]  # same loss for all subheads
    losses = [overcluster_loss_function if i < overcluster_number else normal_loss_function for i in range(subheads_number)]  # diff loss for different heads

    loss_weights_half = [1 / overcluster_number for i in range(
        overcluster_number)]  # weight all subheads equally and reduce importance of a single subheads so that the loss for alle subheads has the same size as only one
    loss_weights = [1 / subheads_number for i in range(
        subheads_number)]

    print("loss weights", loss_weights_half, loss_weights)

    # compile
    model.compile(optimizer=opt, # compile model for callbacks
                  metrics=metrics,
                  loss=losses,
                  loss_weights=loss_weights,
                  )
    for i in range(len(alternate_models)):
        alternate_models[i].compile(optimizer=opt,
                                    metrics=metrics,
                                    loss=normal_losses_half if i == 1 else over_losses_half,
                                    loss_weights=loss_weights_half,
                                    )


    # define generators
    train_gen, mapping_gen, val_gen, extra_gens = get_data_generators_for_key(train_dir, val_dir,
                                                                  target_size, frozen_backbone, config, adaptive_bs=adaptive_bs, extra_names=extra_names, extra_dirs=extra_dirs)

    callbacks = keras_callbacks.callbacks(config, model, mapping_gen, val_gen, extra_names, extra_gens, frozen_backbone=frozen_backbone)

    print("callbacks", callbacks)

    # use validation data only if needed
    if config.use_only_train_cl_acc or config.exclude_intermediate_val:
        _val_gen = None
    else:
        _val_gen = val_gen

    alternate_fit(alternate_models[0], alternate_models[1], model, train_gen, ac, ao, ag,
                    epochs=end_epoch,
                    callbacks=callbacks,
                    use_multiprocessing=True,
                    initial_epoch=start_epoch,
                    validation_data = _val_gen,
                    workers=workers,
                    max_queue_size=10,
                    shuffle=False,
                    verbose=1)





def alternate_fit(model,
                  model2,
                  callback_model,
                  generator,
                  ac, ao, ag,
                  steps_per_epoch=None,
                  epochs=1,
                  verbose=1,
                  callbacks=None,
                  validation_data=None,
                  validation_steps=None,
                  validation_freq=1,
                  class_weight=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0):
    """See docstring for `Model.fit_generator`."""
    epoch = initial_epoch

    do_validation = bool(validation_data)
    model._make_train_function()
    model2._make_train_function()  # added
    if do_validation:
        model._make_test_function()

    # assert do_validation == False # method cant evaluate

    use_sequence_api = is_sequence(generator)
    if not use_sequence_api and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the `keras.utils.Sequence'
                        ' class.'))

    # if generator is instance of Sequence and steps_per_epoch are not provided -
    # recompute steps_per_epoch after each epoch
    recompute_steps_per_epoch = use_sequence_api and steps_per_epoch is None

    if steps_per_epoch is None:
        if use_sequence_api:
            steps_per_epoch = len(generator)
        else:
            raise ValueError('`steps_per_epoch=None` is only valid for a'
                             ' generator based on the '
                             '`keras.utils.Sequence`'
                             ' class. Please specify `steps_per_epoch` '
                             'or use the `keras.utils.Sequence` class.')

    # python 2 has 'next', 3 has '__next__'
    # avoid any explicit version checks
    val_use_sequence_api = is_sequence(validation_data)
    val_gen = (hasattr(validation_data, 'next') or
               hasattr(validation_data, '__next__') or
               val_use_sequence_api)
    if (val_gen and not val_use_sequence_api and
            not validation_steps):
        raise ValueError('`validation_steps=None` is only valid for a'
                         ' generator based on the `keras.utils.Sequence`'
                         ' class. Please specify `validation_steps` or use'
                         ' the `keras.utils.Sequence` class.')

    # Prepare display labels.
    out_labels = callback_model.metrics_names # my: use callback model instead model to get all labels
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

      # prepare callbacks
    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.metrics_names[1:])]
    if verbose:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=model.metrics_names[1:]))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    # callback_model = model._get_callback_model() use the give model

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks._call_begin_hook('train')

    enqueuer = None
    val_enqueuer = None

    try:
        if do_validation:
            if val_gen and workers > 0:
                # Create an Enqueuer that can be reused
                val_data = validation_data
                if is_sequence(val_data):
                    val_enqueuer = OrderedEnqueuer(
                        val_data,
                        use_multiprocessing=use_multiprocessing)
                    validation_steps = validation_steps or len(val_data)
                else:
                    val_enqueuer = GeneratorEnqueuer(
                        val_data,
                        use_multiprocessing=use_multiprocessing)
                val_enqueuer.start(workers=workers,
                                   max_queue_size=max_queue_size)
                val_enqueuer_gen = val_enqueuer.get()
            elif val_gen:
                val_data = validation_data
                if is_sequence(val_data):
                    val_enqueuer_gen = iter_sequence_infinite(val_data)
                    validation_steps = validation_steps or len(val_data)
                else:
                    val_enqueuer_gen = val_data
            else:
                # Prepare data for validation
                if len(validation_data) == 2:
                    val_x, val_y = validation_data
                    val_sample_weight = None
                elif len(validation_data) == 3:
                    val_x, val_y, val_sample_weight = validation_data
                else:
                    raise ValueError('`validation_data` should be a tuple '
                                     '`(val_x, val_y, val_sample_weight)` '
                                     'or `(val_x, val_y)`. Found: ' +
                                     str(validation_data))
                val_x, val_y, val_sample_weights = model._standardize_user_data(
                    val_x, val_y, val_sample_weight)
                val_data = val_x + val_y + val_sample_weights
                if model.uses_learning_phase and not isinstance(K.learning_phase(),
                                                                int):
                    val_data += [0.]
                for cbk in callbacks:
                    cbk.validation_data = val_data

        if workers > 0:
            if use_sequence_api:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    shuffle=shuffle)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if use_sequence_api:
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator

        callbacks.model.stop_training = False
        # ensure both models
        model.stop_training = False
        model2.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            ep_model = model if ac < ao else model2 # added

            print("use model %d" % (1 if ac < ao else 2))
            # ep_model.summary(line_length=200)

            ep_model.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            while steps_done < steps_per_epoch:
                generator_output = next(output_generator)

                if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))

                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
                if x is None or len(x) == 0:
                    # Handle data tensors support when no input given
                    # step-size = 1 for data tensors
                    batch_size = 1
                elif isinstance(x, list):
                    batch_size = x[0].shape[0]
                elif isinstance(x, dict):
                    batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                # build batch logs
                batch_logs = {'batch': batch_index, 'size': batch_size}
                callbacks.on_batch_begin(batch_index, batch_logs)

                outs = ep_model.train_on_batch(x, y,
                                            sample_weight=sample_weight,
                                            class_weight=class_weight,
                                            reset_metrics=False)

                outs = to_list(outs)

                # my ensure only the correct half of the out labels is used
                head = "over" if ac < ao else "norm"
                alternate_out_labels = [l for l in out_labels if head in l or "loss" == l]

                for l, o in zip(alternate_out_labels, outs):
                    batch_logs[l] = o

                callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)

                batch_index += 1
                steps_done += 1

                # Epoch finished.
                if (steps_done >= steps_per_epoch and
                        do_validation and
                        should_run_validation(validation_freq, epoch)):
                    # Note that `callbacks` here is an instance of
                    # `keras.callbacks.CallbackList`
                    if val_gen:
                        val_outs = callback_model.evaluate_generator( # my: set call back model
                            val_enqueuer_gen,
                            validation_steps,
                            callbacks=callbacks,
                            workers=0)
                    else:
                        # No need for try/except because
                        # data has already been validated.
                        val_outs = callback_model.evaluate( # my: set call back model
                            val_x, val_y,
                            batch_size=batch_size,
                            sample_weight=val_sample_weights,
                            callbacks=callbacks,
                            verbose=0)
                    val_outs = to_list(val_outs)
                    # Same labels assumed.
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

                if callbacks.model.stop_training:
                    break

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            ac = (ac + 1) % (ao + ag) # added

            # check if lr was changed by call backs
            callback_lr = K.get_value(callback_model.optimizer.lr)
            _lr = K.get_value(ep_model.optimizer.lr)

            if callback_lr != _lr:
                K.set_value(model.optimizer.lr, callback_lr)
                K.set_value(model2.optimizer.lr, callback_lr)

            if callbacks.model.stop_training:
                break

            if use_sequence_api and workers == 0:
                generator.on_epoch_end()

            if recompute_steps_per_epoch:
                if workers > 0:
                    enqueuer.join_end_of_epoch()

                # recomute steps per epochs in case if Sequence changes it's length
                steps_per_epoch = len(generator)

                # update callbacks to make sure params are valid each epoch
                callbacks.set_params({
                    'epochs': epochs,
                    'steps': steps_per_epoch,
                    'verbose': verbose,
                    'do_validation': do_validation,
                    'metrics': callback_metrics,
                })

    finally:
        try:
            if enqueuer is not None:
                enqueuer.stop()
        finally:
            if val_enqueuer is not None:
                val_enqueuer.stop()

    callbacks._call_end_hook('train')
    return model.history


