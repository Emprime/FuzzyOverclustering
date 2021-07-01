# from src.tensorflow_keras import *
from os.path import join

import numpy as np
from keras import Input, Model
from keras.engine.saving import load_model
from keras.initializers import Initializer
from keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Dense, Concatenate
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.utils import multi_gpu_model

from src.archs.resnet_common import ResNet50V2
from src.archs.resnet_variant import ResNet34 as ResNet34Variant

from src.utils.losses import  combined_loss, triplet_loss,  metric_accuracy, metric_ig_first, metric_ig_second, metric_ce, metric_ce_inv
from src.utils import util, const




def weights_and_compile(config, model, model_withouth_bottom, alternate_models, before_frozen, restart, continueWeights,  frozen_backbone, weight_path = None):
    """
    load weights and compile
    :param model: model to use
    :param model_withouth_bottom: model without bottom to use for pretraining, used here due to previous freezing and unfreezung
    :param losses: array of losses to use during model compile
    :param config: config file
    :param frozen_backbone: defines if the backbone should have frozen weights
    :param before_frozen: need to load weights into a frozen model if it was frozen before
    :param weight_path: use a specific weight_path instead of former experiment
    :return: the specified model
    """

    experiment_identifiers = config.experiment_identifiers
    sub_experiment_name = config.sub_experiment_name
    sobel = config.sobel

    if sobel:
        # ensure layer is frozen for restart
        model_withouth_bottom.layers[1].trainable = False

    if before_frozen:
        # need to load weights into a frozen model if it was frozen before
        for layer in model_withouth_bottom.layers[:-1]:
            layer.trainable = False

    if restart or continueWeights:
        if weight_path is None or continueWeights:
            weight_path = util.get_weights_latest(config.root_log_dir,experiment_identifiers, sub_experiment_name, ignore_current=restart) # ignore current only in restart
        print("load weights from %s" % weight_path)
        model.load_weights(weight_path)

    if before_frozen:
        # need to load weights into a frozen model if it was frozen before
        for layer in model_withouth_bottom.layers[:-1]:
            layer.trainable = True

    # frozen model only stores not frozen weights
    if frozen_backbone:
        for layer in model_withouth_bottom.layers[:-1]:
            layer.trainable = False


    if sobel:
        # print( model_withouth_bottom.layers[1])
        # print("not trainable")
        model_withouth_bottom.layers[1].trainable = False

    model.summary(line_length=200)


    if config.num_gpus > 1:
        print("use multi-gpu model")
        model = multi_gpu_model(model, gpus=config.num_gpus)
        for i in range(len(alternate_models)):
            alternate_models[i] = multi_gpu_model( alternate_models[i], gpus=config.num_gpus)


    # save graph
    config.graph = tf.get_default_graph()

    return model, alternate_models


class sobel_input_kernel( Initializer ):
  def __init__(self):
      pass

  def __call__(self, shape, dtype=None):
      # print(shape)
      kernel = np.zeros(shape, dtype=dtype)

      # decide only sobel or sobel and rgb
      channels = shape[3]

      # sobel
      kernel[:, :, 0, 0] = 0.3 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # sobel x
      kernel[:, :, 1, 0] = 0.59 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # sobel x
      kernel[:, :, 2, 0] = 0.11 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # sobel x
      kernel[:, :, 0, 1] = 0.3 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # sobel y
      kernel[:, :, 1, 1] = 0.59 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # sobel y
      kernel[:, :, 2, 1] = 0.11 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # sobel y

      if channels == 5:  # add rgb
          # identity
          kernel[1, 1, 0, 2] = 1
          kernel[1, 1, 1, 3] = 1
          kernel[1, 1, 2, 4] = 1

      return kernel

  def get_config(self):
      return {}


def get_model_with_architecture(architecture):
    def get_model(config, frozen_backbone=True, before_frozen=True, restart=False, continueWeights=False, use_input=None):
        """
        get the model for iic clustering
        :param config: config file
        :param frozen_backbone: defines if the backbone should have freezed weights
        :param before_frozen: need to load weights into a freezed model if it was freezed before
        :param return_only_model_without_bottom: hacky way to circumvent an error in keras
        :return: the specified model
        """

        cluster_outputs = config.overcluster_k
        gt_k = config.gt_k
        pretrained = config.pretrained
        dropout = config.dropout
        overcluster_number = config.subheads_number
        gt_number = config.subheads_gt_number
        subheads_number = overcluster_number + gt_number
        weight_path = config.weight_path
        input_size = config.input_size
        sobel_only = config.sobel_only
        sobel_rgb = config.sobel_rgb
        dropout_percent = config.dropout_percent




        # shared model
        input_shape = (input_size, input_size, 3)
        input_tensor = Input(shape=input_shape)

        if sobel_only or sobel_rgb:

            channels = (5 if sobel_rgb else 2)
            start = Conv2D(name="sobel", filters=channels, kernel_size=(3, 3),
                           kernel_initializer=sobel_input_kernel(), bias_initializer='zeros', padding='same')(
                input_tensor)
        else:
            start = input_tensor

        if use_input is not None:
            # needed to fix bug in keras with double nested models
            input_tensor = use_input
            start = use_input





        # create new model
        if architecture == "resnet":
            model_withouth_bottom = ResNet34Variant(include_top=False, input_tensor=start)
        elif architecture == "resnet50v2":
            model_withouth_bottom = ResNet50V2(include_top=False, weights='imagenet' if pretrained else None, input_tensor=start)
        else:
            raise KeyError("architecture %s not available" % architecture)


        subhead_outputs = []
        overcluster_outputs = []
        normal_outputs = []
        # if not fine_tune or return_only_model_without_bottom:

        x = model_withouth_bottom.output
        x = GlobalAveragePooling2D(name="eoc_global_ap")(x)
        if dropout:
            x = Dropout(dropout_percent)(x)

        # add subheads / outputs

        # normal subheads
        for i in range(overcluster_number):
            clusters = Dense(cluster_outputs, activation='softmax')(x)
            subhead_outputs.append(clusters)
            overcluster_outputs.append(clusters)
        # gt cluster size subheads
        for i in range(gt_number):
            clusters = Dense(gt_k, activation='softmax')(x)
            subhead_outputs.append(clusters)
            normal_outputs.append(clusters)



        shared_model = Model(inputs=input_tensor, outputs=subhead_outputs, name='shared-base-model-with-subheads')


        # MERGE inputs
        input1 = Input(shape=input_shape, name='input1')
        input2 = Input(shape=input_shape, name='input2')
        input3 = Input(shape=input_shape, name='input3')

        subhead_outputs_1 = shared_model(input1)
        subhead_outputs_2 = shared_model(input2)
        subhead_outputs_3 = shared_model(input3)

        # define alternate models
        alternate_models = []

        # overcluster model
        m = Model(inputs=input_tensor, outputs=overcluster_outputs, name='shared-base-model-with-overcluster-heads')
        merged_output_heads = []
        if overcluster_number == 1:
            merged = Concatenate(name="output_head")([m(input1), m(input2), m(input3)])
            merged_output_heads.append(merged)
        else:
            for i in range(overcluster_number):
                merged = Concatenate(name="over-head-%d" % i)([m(input1)[i], m(input2)[i], m(input3)[i]])
                merged_output_heads.append(merged)

        alternate_models.append(Model(inputs=[input1, input2, input3], outputs=merged_output_heads))

        # normal model
        m = Model(inputs=input_tensor, outputs=normal_outputs,
                  name='shared-base-model-with-normal-heads')
        merged_output_heads = []
        if gt_number == 1:
            merged = Concatenate(name="output_head")([m(input1), m(input2), m(input3)])
            merged_output_heads.append(merged)
        else:
            for i in range(gt_number):
                merged = Concatenate(name="norm-head-%d" % i)([m(input1)[i], m(input2)[i], m(input3)[i]])
                merged_output_heads.append(merged)

        alternate_models.append(Model(inputs=[input1, input2, input3], outputs=merged_output_heads))


        merged_output_heads = []

        if subheads_number == 1:
            merged = Concatenate(name="output_head")([subhead_outputs_1, subhead_outputs_2,subhead_outputs_3])
            merged_output_heads.append(merged)

        else:
            for i in range(subheads_number):
                is_overcluster = i < overcluster_number
                head_name = "%s-head-%d" % (
                "over" if is_overcluster else "norm", i if is_overcluster else i - overcluster_number)
                merged = Concatenate(name=head_name)([subhead_outputs_1[i], subhead_outputs_2[i],subhead_outputs_3[i]])
                merged_output_heads.append(merged)


        model = Model(inputs=[input1, input2, input3], outputs=merged_output_heads)

        model, alternate_models = weights_and_compile(config, model,model_withouth_bottom, alternate_models, before_frozen, restart, continueWeights, frozen_backbone, weight_path=weight_path)

        return model, alternate_models
    return get_model