import sys
from sklearn import metrics
import numpy as np
from tensorflow.python.keras import backend as K
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.optimize import linear_sum_assignment as linear_assignment
from keras.metrics import categorical_accuracy


def ce_loss(y_true,y_pred):
    """
    own crosse etnropy loss implementation
    :param y_true:
    :param y_pred:
    :return:
    """
    # seems to be identical to keras loss
    y = K.clip(y_true, K.epsilon(),1-K.epsilon())
    x = K.clip(y_pred, K.epsilon(),1-K.epsilon())

    ce = -K.sum(y * K.log(x), axis=1)

    return ce

def kl_div(y_true,y_pred):
    """
       own kullback  loss implementation
       :param y_true:
       :param y_pred:
       :return:
    """
    # seems to be identical to keras loss
    y = K.clip(y_true, K.epsilon(),1-K.epsilon())
    x = K.clip(y_pred, K.epsilon(),1-K.epsilon())


    ce = K.sum(y * K.log(y/x), axis=1)

    return ce

def inverse_ce(y_true, y_pred):
    """
    new loss for foc frametwork
    :param y_true:
    :param y_pred:
    :return:
    """
    y = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
    x = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # x_inv = K.softmax(1-x)
    x_inv = 1-x

    inv_ce = -K.sum(y * K.log(x_inv), axis=1)

    return inv_ce




def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)



def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """
    copied from iid / iic paper
    :param x_out:
    :param x_tf_out:
    :param lamb:
    :param EPS:
    :return:
    """
    p_i_j, p_i, p_j = compute_P_matrices(x_out,x_tf_out, EPS)
    loss = - p_i_j * (K.log(p_i_j) - lamb * K.log(p_j) - lamb * K.log(p_i))
    loss = K.sum(loss)
    return loss

def compute_P_matrices(x_out, x_tf_out, EPS):
    """
    copied from iid / iic paper
    :param x_out:
    :param x_tf_out:
    :return:
    """
    # has had softmax applied
    shape = K.shape(x_out)
    k = shape[1]
    # print(K.eval(k))
    p_i_j = compute_joint(x_out, x_tf_out)

    p_i = tf.broadcast_to(K.expand_dims(K.sum(p_i_j, axis=1), axis=1), [k, k])
    p_j = tf.broadcast_to(K.expand_dims(K.sum(p_i_j, axis=0), axis=0), [k, k])  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j = K.clip(p_i_j, EPS, None)
    p_i = K.clip(p_i, EPS, None)
    p_j = K.clip(p_j, EPS, None)

    return  p_i_j, p_i, p_j


def compute_joint(x_out, x_tf_out):
    """
    copied from iid/iic paper
    :param x_out:
    :param x_tf_out:
    :return:
    """
    # produces variable that requires grad (since args require grad)

    # expand dimensions and multiply
    p_i_j = K.expand_dims(x_out, axis=2) * K.expand_dims(x_tf_out, axis=1)  # bn, k, k
    p_i_j = K.sum(p_i_j, axis=0)  # k, k
    p_i_j = (p_i_j + K.transpose(p_i_j)) / 2.  # symmetrise
    p_i_j = p_i_j / K.sum(p_i_j)  # normalise

    return p_i_j

def cast_pred_to_x_y(y_pred):
    """
    split the prediction into two halfs
    :param y_pred:
    :return:
    """

    nb_outputs = K.shape(y_pred)[1] // 2

    x = y_pred[:, :nb_outputs]
    y = y_pred[:, nb_outputs:]


    return x,y

def cast_pred_to_x_y_z(y_pred):
    """
    split the prediction into three thirds
    :param y_pred:
    :return:
    """

    nb_outputs = K.shape(y_pred)[1] // 3

    x = y_pred[:, :nb_outputs]
    y = y_pred[:, nb_outputs:2*nb_outputs]
    z = y_pred[:, 2*nb_outputs:]


    return x,y,z


def metric_ce_inv(use_triplet, lambda_t):
    def ce_inv(y_true, y_pred, EPS=sys.float_info.epsilon):
        """
        cast output so that first half of IID loss can be caluclated, first half is entropy over x,
        the sum of first and secound should match info_gain_loss
        :param y_true:
        :param y_pred:
        :return:
        """

        if use_triplet:
            x,y,z = cast_pred_to_x_y_z(y_pred)

            triplet1 = inverse_ce(x, z)
            triplet2 = inverse_ce(y, z)

            triplet_loss = (0.5 * (triplet1 + triplet2))
        else:
            x, y = cast_pred_to_x_y(y_pred)

            triplet_loss = inverse_ce(x,y)


        return lambda_t * triplet_loss
    return ce_inv


def metric_ig_first(use_triplet, lambda_m):
    def ig_first(y_true, y_pred, EPS=sys.float_info.epsilon):
        """
        cast output so that first half of IID loss can be caluclated, first half is entropy over x,
        the sum of first and secound should match info_gain_loss
        :param y_true:
        :param y_pred:
        :return:
        """

        if use_triplet:
            x,y,z = cast_pred_to_x_y_z(y_pred)
        else:
            x, y = cast_pred_to_x_y(y_pred)

        p_i_j, p_i, p_j = compute_P_matrices(x, y, EPS)

        loss = lambda_m * K.sum(- p_i_j * K.log(p_i)) # entropy of p_i but over iteration of i,j
        return - loss # maximize return - H(x)
    return ig_first


def metric_ig_second(use_triplet, lambda_m):
    def ig_secound(y_true, y_pred, EPS=sys.float_info.epsilon):
        """
        cast output so that sec half of IID loss can be caluclated, sec half is cond. entropy of H(x|y)
         the sum of first and secound should match info_gain_loss
        :param y_true:
        :param y_pred:
        :return:
        """

        if use_triplet:
            x,y,z = cast_pred_to_x_y_z(y_pred)
        else:
            x, y = cast_pred_to_x_y(y_pred)

        p_i_j, p_i, p_j = compute_P_matrices(x, y, EPS)

        loss = lambda_m * K.sum(- p_i_j * (K.log(p_i_j) - K.log(p_j) ))  #conditional entropy H(p_i  | p_j)
        return loss # maximize return - (-H(x|y)
    return ig_secound

def metric_accuracy(use_triplet):
    def acc(y_true,y_pred):

        if use_triplet:
            x,y,z = cast_pred_to_x_y_z(y_pred)
        else:
            x, y = cast_pred_to_x_y(y_pred)

        return categorical_accuracy(y_true,x)
    return acc




def metric_ce(use_triplet,lambda_s):
    def loss_ce_heads(y_true, y_pred):

        if use_triplet:
            x,y,z = cast_pred_to_x_y_z(y_pred)
        else:
            x, y = cast_pred_to_x_y(y_pred)

        supervised_loss = lambda_s * ce_loss(y_true, x)

        return supervised_loss
    return loss_ce_heads


def combined_loss(lambda_s, lambda_m = 1):
    """

    :param lambda_s: supervised loss weight
    :param lambda_e: entropy loss weight
    :param lambda_c: consistency loss weight
    :return:
    """

    def combined_internal(y_true, y_pred, EPS=sys.float_info.epsilon):
        x, y = cast_pred_to_x_y(y_pred)
        p_i_j, p_i, p_j = compute_P_matrices(x, y, EPS)


        supervised_loss = lambda_s * ce_loss(y_true, x)
        # supervised_loss = supervised_loss *0.5 + 0.5 * (lambda_s * ce_loss(y_true, y))
        entropy_loss = lambda_m * -K.sum(- p_i_j * K.log(p_i)) # entropy of p_i but over iteration of i,j
        consistency_loss = lambda_m * K.sum(- p_i_j * (K.log(p_i_j) - K.log(p_j)))

        return supervised_loss + entropy_loss + consistency_loss

    return combined_internal

def triplet_loss(lambda_s,lambda_t, lambda_m= 1):
    """
    identical to combined loss but uses triplet
    :param lambda_s: supervised loss weight
    :param lambda_e: entropy loss weight
    :param lambda_c: consistency loss weight
    :return:
    """

    def combined_internal(y_true, y_pred, EPS=sys.float_info.epsilon):

        x, y, z = cast_pred_to_x_y_z(y_pred)


        p_i_j, p_i, p_j = compute_P_matrices(x, y, EPS)

        # loss per input triplet
        supervised_loss = lambda_s * ce_loss(y_true, x)

        # loss per batch
        entropy_loss = lambda_m * -K.sum(- p_i_j * K.log(p_i)) # entropy of p_i but over iteration of i,j
        consistency_loss = lambda_m * K.sum(- p_i_j * (K.log(p_i_j) - K.log(p_j)))

        # triplet loss, per input triplet
        triplet1 = inverse_ce(x, z)
        triplet2 = inverse_ce(y, z)

        triplet_loss_part = lambda_t * (0.5 * (triplet1 + triplet2)) # average over triplet


        # calculate ce inv triplet loss only for labeled data points
        triplet_loss_part = K.mean(y_true, axis=1) * triplet_loss_part

        return supervised_loss + entropy_loss + consistency_loss + triplet_loss_part # + g

    return combined_internal


def NMI(y_true,y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred,average_method='arithmetic')

def ACC(y_true,y_pred):
    """
    expects int labels, accuracy with best mapping between true and pred
    :param y_true:
    :param y_pred:
    :return:
    """
    assert np.max(y_true) >= np.max(y_pred) # not more predictated than available

    Y_pred = y_pred
    Y = y_true

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = np.transpose(ind)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind

def PURITY(y_true,y_pred, ignore_noise=False, return_per_cluster=False):
    """
    It is the percent of the total number of objects(data points) that were classified correctly, in the unit range [0..1].
    :param y_true:
    :param y_pred:
    :return:
    """

    assert len(y_true) == len(y_pred)

    if ignore_noise:
        # ignore noise clusters
        y_true = y_true[y_pred != -1]
        y_pred = y_pred[y_pred != -1]

    if len(y_true) == 0:
        # only noise in array or empty
        return 1 # empty is pure

    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    clusters, _ = np.unique(y_pred, return_inverse=True)

    pur = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    if return_per_cluster:
        purity_per_cluster = np.amax(contingency_matrix, axis=0) / np.sum(contingency_matrix,axis=0)

        return pur, dict(zip(clusters,purity_per_cluster))
    else:
        return pur


def Combine_clusters_by_purity(y_true,y_pred, return_mapping = False, mapping=None):
    """
    combine more clusters to number of clusters of gt, use only first n examples for gt propagation
    :param y_true:
    :param y_pred:
    :return:
    """
    # filter for both valid values
    y_true_masked, y_pred_masked, mask = mask_Nones(y_true, y_pred, return_mask=True)

    contingency_matrix = metrics.cluster.contingency_matrix(y_true_masked, y_pred_masked)

    # used for indexing if y-true and y-pred are not containing values like [0,1,2,3,..]
    classes, class_idx = np.unique(y_true_masked, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred_masked, return_inverse=True)

    if mapping is None:

        raw_mapping = np.argmax(contingency_matrix, axis=0)

        # calculate mapping
        mapping = []
        for x, y in enumerate(raw_mapping):
            mapping.append((clusters[x],classes[y]))

    result = np.zeros(y_pred_masked.shape)
    unmapped_clusters = list(np.copy(clusters))
    for x,y in mapping:
        np.putmask(result,y_pred_masked == x,y)
        if x in unmapped_clusters:
            unmapped_clusters.remove(x)

    for unmapped_cluster in unmapped_clusters:
        np.putmask(result, y_pred_masked == unmapped_cluster, -1)

    # unmask
    temp = y_pred.copy()
    temp[mask] = result.astype(int)
    result = temp


    if return_mapping:
        return result, mapping
    else:
        return result

def mask_Nones(dist_x, dist_y, file_names=None, return_mask=False, return_number=False):
    """
    mask the elements which are none in dist x or dist y
    :param dist_x:
    :param dist_y:
    :param file_names: return the file names corresponding to the dists, will return as last value the file names
    :param return_mask:
    :param return_number:
    :return:
    """
    if return_number:
        number = -1
        dist_x = dist_x.copy()
        dist_y = dist_y.copy()
        dist_x[dist_x == None] = number
        dist_y[dist_y == None] = number

        if file_names is None:
            return dist_x.astype(int), dist_y.astype(int)
        else:
            return dist_x.astype(int), dist_y.astype(int), file_names.copy()

    mask = (dist_x != None) & (dist_y != None)
    dist_x = dist_x[mask].astype(int)
    dist_y = dist_y[mask].astype(int)

    if file_names is not None:
        file_names = np.array(file_names)[mask]

        if return_mask:
            return dist_x, dist_y, mask, file_names
        else:
            return dist_x, dist_y, file_names
    else:

        if return_mask:
            return dist_x, dist_y, mask
        else:
            return dist_x, dist_y