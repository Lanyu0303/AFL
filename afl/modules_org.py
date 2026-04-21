# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import tensorflow as tf


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
    return outputs


def embedding(
    inputs,
    vocab_size,
    num_units,
    zero_pad=True,
    scale=True,
    l2_reg=0.0,
    scope="embedding",
    with_t=False,
    reuse=None,
):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        lookup_table = tf.compat.v1.get_variable(
            "lookup_table",
            dtype=tf.float32,
            shape=[vocab_size, num_units],
            regularizer=tf.keras.regularizers.l2(l2_reg),
        )
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    return outputs


def multihead_attention(
    queries,
    keys,
    num_units=None,
    num_heads=8,
    dropout_rate=0,
    is_training=True,
    causality=False,
    scope="multihead_attention",
    reuse=None,
    with_qk=False,
):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        Q = tf.compat.v1.layers.dense(queries, num_units, activation=None)
        K = tf.compat.v1.layers.dense(keys, num_units, activation=None)
        V = tf.compat.v1.layers.dense(keys, num_units, activation=None)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2**32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            paddings = tf.ones_like(masks) * (-2**32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)

        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks

        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries

    if with_qk:
        return Q, K
    return outputs


def feedforward(
    inputs,
    num_units=(2048, 512),
    scope="multihead_attention",
    dropout_rate=0.2,
    is_training=True,
    reuse=None,
):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        outputs += inputs
    return outputs
