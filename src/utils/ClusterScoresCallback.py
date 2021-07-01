import keras
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from src.utils.losses import ACC, Combine_clusters_by_purity


class ClusterScoresCallbacks(keras.callbacks.Callback):

    def __init__(self, datasplit_name, generator, model, config, training_callback=None, use_graph=True):
        super().__init__()

        print("init cluster score callback %s" % datasplit_name)

        self.datasplit_name = datasplit_name
        self.generator = generator

        self.graph = config.graph
        self.worker = config.workers * config.num_gpus
        self.config = config

        self.log_dir = config.log_dir

        # setup the cluster extraction
        self.my_model = model
        self.subheads_number = generator.subheads_number  # len(cluster_extractors)
        self.overcluster_subheads_number = generator.overcluster_subheads_number  # len(cluster_extractors)
        self.gt_subheads_number = generator.gt_subheads_number  # len(cluster_extractors)
        self.logging_step = config.cluster_prediction_step  # determines how often the clusterer is used
        self.logging_count = 0

        self.use_graph = use_graph

        # used for mapping calculation based on training
        self.training_callback = training_callback
        self.training_mappings = [None for i in range(
            self.subheads_number)] if training_callback is None else None  # only used if training callback is None


        self.scores = {}


    def predict_cluster_scores(self, epoch):
        """
        calculate cluster scores based on the prediction, use cluster-acc, nmi and purity
        :return:
        """


        print("%s - evaluate on all heads and datapoints" % self.datasplit_name)


        if self.use_graph:

            with self.graph.as_default():
                all_preds = self.my_model.predict_generator(
                self.generator,
                use_multiprocessing=True,
                workers=self.worker,
                max_queue_size=10,
                verbose=1)
        else:
            # bug in keras, loses workers otherwiese
            all_preds = self.my_model.predict_generator(
                self.generator,
                use_multiprocessing=False,
                workers=1,
                max_queue_size=10,
                verbose=1)


        gt = self.generator.labels
        file_names = self.generator.file_names

        for i in tqdm(range(self.subheads_number)):


            # get description of head
            pred_head = np.array(all_preds[i] if self.subheads_number > 1 else all_preds)
            # print("predict head", pred_head.shape)

            overcluster_subheads_number = self.generator.overcluster_subheads_number
            if i < overcluster_subheads_number:
                preds = np.argmax(pred_head[:, :self.generator.overcluster_num_classes], axis=1)
                type = "over"
                num = i

                # try to use cluster accuracy
                try:
                    _, mapping = ACC(gt, preds)  # fails if more clusters than gt classes
                    mapped_pred = Combine_clusters_by_purity(gt, preds, mapping=mapping)
                except:
                    # calculate mapping and predictions
                    mapped_pred, mapping = Combine_clusters_by_purity(gt, preds, return_mapping=True)

            else:
                preds = np.argmax(pred_head[:, :self.generator.num_classes],axis=1)
                type = "normal"
                num = i - overcluster_subheads_number

                # classification no mapping required
                mapped_pred = preds

            suffix = "%s%d" % (type, num)



            cl_report = metrics.classification_report(gt, mapped_pred, digits=4, output_dict=True)

            self.scores["%s_%s_%s" % (self.datasplit_name, "macro_f1", suffix)] =  cl_report['macro avg']['f1-score']

            self.scores["%s_%s_%s" % (self.datasplit_name, "cluster_acc", suffix)]  = metrics.accuracy_score(gt,
                                                                              mapped_pred)

        return all_preds





    def on_epoch_end(self, epoch, logs=None, force_logging = False):
        'check if predictions should be made'

        # decide to use cluster extractor
        if (self.model is not None
                and self.logging_count % self.logging_step == 0
                and self.logging_step > 0) or force_logging:
            all_preds = self.predict_cluster_scores(epoch)
        else:
            all_preds = None


        self.logging_count += 1

        return all_preds
