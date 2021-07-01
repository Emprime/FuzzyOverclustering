from src.data_loaders.multi_image_loader import ImageLoader
from src.utils import util


def get_data_generators_for_key(train_dir, val_dir, target_size, frozen_backbone,  config,extra_names = [], extra_dirs=[],adaptive_bs=False):

    params = {
        'batch_size':config.frozen_batch_size if frozen_backbone else config.batch_size ,
        'target_size':target_size,
        'loading_size':config.loading_size,
        'overcluster_subheads_number': config.subheads_number,
        'gt_subheads_number': config.subheads_gt_number,
        'overcluster_num_classes': config.overcluster_k,
        'num_classes' : config.gt_k,
        'alternate_overcluster_max' : config.alternate_overcluster_max,
        'alternate_gt_max': config.alternate_gt_max ,
        'max_percent_unlabeled_data_in_batch' : config.max_percent_unlabeled_data_in_batch,

    }



    train_params = {**params, **{'imgaug_augments': [config.imgaug_0, config.imgaug_1, config.imgaug_2],
                                 'shuffle':config.shuffle,'unlabeled_data_dir' : config.unlabeled_data_dir, 'alternate_count' : 0,
                                 'sample_repetition': config.sample_repetition,  'adaptive_bs':False,}}
    val_params = {**params, **{'imgaug_augments': ['val_crop','val_crop','val_crop'],
                               'shuffle':False,'unlabeled_data_dir' : None, 'alternate_count' : -1,
                               'sample_repetition': 1,  'adaptive_bs':adaptive_bs,}}

    train_gen = ImageLoader("train gen", train_dir, **train_params)

    mapping_gen = ImageLoader("mapping gen", train_dir, **val_params)
    val_gen = ImageLoader("val gen", val_dir, **val_params)

    # use extra datagenerators for more predictions
    assert len(extra_names) == len(extra_dirs)
    extra_gens = []
    for name, dir in zip(extra_names,extra_dirs):
        gen = ImageLoader(name, dir, **val_params)
        extra_gens.append(gen)

    return train_gen, mapping_gen, val_gen, extra_gens


