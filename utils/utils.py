import tensorflow as tf
from tensorflow.keras import backend as K


def lr_multiply_ratio(initial_lr, lr_ratio):
    def lr_multiplier(idx):
        return initial_lr*lr_ratio**idx
    return lr_multiplier


def wln_loss(y_true, y_pred):

    #softmax cross entropy
    flat_label = K.cast(K.reshape(y_true, [-1]), 'float32')
    flat_score = K.reshape(y_pred, [-1])

    reaction_seg = K.cast(tf.math.cumsum(flat_label), 'int32') - tf.constant([1], dtype='int32')

    max_seg = tf.gather(tf.math.segment_max(flat_score, reaction_seg), reaction_seg)
    exp_score = tf.exp(flat_score-max_seg)

    softmax_denominator = tf.gather(tf.math.segment_sum(exp_score, reaction_seg), reaction_seg)
    softmax_score = exp_score/softmax_denominator

    softmax_score = tf.clip_by_value(softmax_score, K.epsilon(), 1-K.epsilon())
    try:
        return -tf.reduce_sum(flat_label * tf.math.log(softmax_score))/flat_score.shape[0]
    except:
        #during initialization
        return -tf.reduce_sum(flat_label * tf.math.log(softmax_score))


def regio_acc(y_true_g, y_pred):
    y_true_g = K.reshape(y_true_g, [-1])
    y_pred = K.reshape(y_pred, [-1])

    reaction_seg = K.cast(tf.math.cumsum(y_true_g), 'int32') - tf.constant([1], dtype='int32')

    top_score = tf.math.segment_max(y_pred, reaction_seg)
    major_score = tf.gather(y_pred, tf.math.top_k(y_true_g, tf.size(top_score))[1])

    match = tf.equal(top_score, major_score)
    return tf.reduce_sum(K.cast(match, 'float32'))/K.cast(tf.size(top_score), 'float32')