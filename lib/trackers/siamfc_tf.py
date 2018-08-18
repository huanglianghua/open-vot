from __future__ import absolute_import, division

import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib import slim
from collections import namedtuple
from PIL import Image

from . import Tracker
from ..utils.viz import show_frame


class AlexNet(object):

    def __init__(self, **kargs):
        self.parse_args(**kargs)

    def __call__(self, inputs, trainable=True,
                 is_training=False, reuse=False):
        arg_scope = self.setup_argscope(trainable, is_training)
        with slim.arg_scope(arg_scope):
            return self.inference(inputs, reuse)

    def parse_args(self, **kargs):
        self.cfg = {
            'bn_scale': True,
            'bn_momentum': 0.05,
            'bn_epsilon': 1e-6,
            'weight_decay': 5e-4,
            'init_method': 'kaiming_normal'}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

    def setup_argscope(self, trainable=False, is_training=False):
        # batchnorm parameters
        norm_params = {
            'scale': self.cfg.bn_scale,
            'decay': 1 - self.cfg.bn_momentum,
            'epsilon': self.cfg.bn_epsilon,
            'trainable': trainable,
            'is_training': trainable and is_training,
            'variables_collections': {
                'beta': None,
                'gamma': None,
                'moving_mean': ['moving_vars'],
                'moving_variance': ['moving_vars']},
            'updates_collections': None}
        norm_fn = slim.batch_norm

        # regularizer
        if trainable:
            weights_regularizer = slim.l2_regularizer(self.cfg.weight_decay)
        else:
            weights_regularizer = None

        # initializer
        if self.cfg.init_method == 'kaiming_normal':
            # the same setting as siamese-fc
            weights_initializer = slim.variance_scaling_initializer(
                factor=2.0, mode='FAN_OUT', uniform=False)
        else:
            weights_initializer = slim.xavier_initializer()

        # setup argument scope
        with slim.arg_scope(
            # conv2d
            [slim.conv2d],
            weights_regularizer=weights_regularizer,
            weights_initializer=weights_initializer,
            padding='VALID',
            trainable=trainable,
            activation_fn=tf.nn.relu,
            normalizer_fn=norm_fn,
            normalizer_params=norm_params):
            with slim.arg_scope(
                # batchnorm
                [slim.batch_norm],
                **norm_params):
                with slim.arg_scope(
                    # batchnorm
                    [slim.batch_norm],
                    is_training=trainable and is_training) as arg_scope:
                    return arg_scope

    def inference(self, inputs, reuse=False):
        with tf.variable_scope(
                'convolutional_alexnet', [inputs], reuse=reuse) as sc:
            end_points_collections = sc.name + '_end_points'
            with slim.arg_scope(
                [slim.conv2d, slim.max_pool2d],
                outputs_collections=end_points_collections):
                # 0: input
                net = inputs
                # 1: conv1
                net = slim.conv2d(net, 96, [11, 11], 2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
                # 2: conv2
                with tf.variable_scope('conv2'):
                    # group convolution
                    b1, b2 = tf.split(net, 2, 3)
                    b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
                    b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
                    net = tf.concat([b1, b2], 3)
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
                # 3: conv3
                net = slim.conv2d(net, 384, [3, 3], 1, scope='conv3')
                # 4: conv4
                with tf.variable_scope('conv4'):
                    # group convolution
                    b1, b2 = tf.split(net, 2, 3)
                    b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
                    b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
                    net = tf.concat([b1, b2], 3)
                # 5: conv5
                with tf.variable_scope('conv5'):
                    with slim.arg_scope(
                        [slim.conv2d],
                        activation_fn=None,
                        normalizer_fn=None):
                        # group convolution
                        b1, b2 = tf.split(net, 2, 3)
                        b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
                        b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
                    net = tf.concat([b1, b2], 3)
                # convert outputs collections to a dictionary
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collections)

                return net, end_points


class GraphSiamFC(object):

    def __init__(self, **kagrs):
        self.parse_args(**kagrs)
        if self.cfg.trainable:
            # graph input: filenames_z, boxes_z, filenames_x, boxes_x
            # graph output: loss
            self.setup_train_graph()
        else:
            # graph input: filename, box
            # graph output: init_op, response
            self.setup_inference_graph()
        # setup saver for saving to/restoring from checkpoints
        self.setup_saver()

    def parse_args(self, **kargs):
        self.cfg = {
            'trainable': False,
            'exemplar_sz': 127,
            'search_sz': 255,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_up': 8,
            'adjust_scale': 0.001}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

    def setup_train_graph(self):
        # placeholders
        filenames_z = tf.placeholder(tf.string, [None], name='filenames_z')
        filenames_x = tf.placeholder(tf.string, [None], name='filenames_x')
        boxes_z = tf.placeholder(tf.float32, [None, 4], name='boxes_z')
        boxes_x = tf.placeholder(tf.float32, [None, 4], name='boxes_x')

        # transformations
        def transform_fn(pairs):
            pass
        exemplars, instances = tf.map_fn(
            transform_fn, (filenames_z, filenamex_x, boxes_z, boxes_x),
            dtype=(tf.float32, tf.float32))

        # embedding
        net = AlexNet()
        exemplar_embeds = net(exemplars, trainable=True, is_training=True)
        instance_embeds = net(instances, trainable=True, is_training=True)

        # responses
        with tf.variable_scope('detection'):
            def xcorr(x, z):
                x = tf.expand_dims(x, 0)
                z = tf.expand_dims(z, -1)
                return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1],
                                    padding='VALID', name='xcorr')

            output = tf.map_fn(
                lambda x: xcorr(x[0], x[1]),
                (search_embeds, templates),
                dtype=search_embeds.dtype)
            output = tf.squeeze(output, [1, 4])

            bias = tf.get_variable(
                'biases', [1], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                trainable=True)
            response = self.cfg.adjust_scale * output + bias

        # loss
        labels = self.setup_labels(response.get_shape().as_list()[1:])
        with tf.name_scope('loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=response, labels=labels)
            with tf.name_scope('balance_weights'):
                pos_num = tf.reduce_sum(tf.to_float(tf.equal(labels[0], 1)))
                neg_num = tf.reduce_sum(tf.to_float(tf.equal(labels[0], 0)))
                pos_weight = 0.5 / pos_num
                neg_weight = 0.5 / neg_num
                weights = tf.where(
                    tf.equal(labels, 1),
                    pos_weight * tf.ones_like(labels),
                    tf.ones_like(labels))
                weights = tf.where(
                    tf.equal(labels, 0),
                    neg_weight * tf.ones_like(labels),
                    weights)
                loss = loss * weights
            loss = tf.reduce_sum(loss, axis=(1, 2))

            self.batch_loss = tf.reduce_mean(loss, name='batch_loss')
            tf.losses.add_loss(self.batch_loss)
            self.total_loss = tf.losses.get_total_loss()

    def setup_inference_graph(self):
        # placeholders
        filename = tf.placeholder(tf.string, [], name='filename')
        box = tf.placeholder(tf.float32, [4], name='box')

        # convert box to 0-indexed and center based [y, x, h, w]
        box = tf.stack([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]])
        center, target_sz = box[:2], box[2:]

        # inputs
        image = tf.image.decode_jpeg(tf.read_file(
            filename), channels=3, dct_method='INTEGER_ACCURATE')
        image = tf.to_float(image)
        image_sz = tf.shape(image)

        # exemplar and search sizes
        context = self.cfg.context * tf.reduce_sum(target_sz)
        z_sz = tf.sqrt(tf.reduce_prod(target_sz + context))
        x_sz = z_sz * self.cfg.search_sz / self.cfg.exemplar_sz

        # multi-scale searching bounding boxes
        scales = np.arange(self.cfg.scale_num) - self.cfg.scale_num // 2
        assert np.sum(scales) == 0, 'scale_num should be an odd number'
        self.scale_factors = self.cfg.scale_step ** scales
        boxes = []
        for factor in self.scale_factors:
            scaled_search_area = factor * x_sz
            image_sz_1 = tf.to_float(image_sz[:2] - 1)
            boxes.append(tf.concat([
                tf.div(center - (x_sz - 1) / 2, image_sz_1),
                tf.div(center + (x_sz - 1) / 2, image_sz_1)], axis=0))
        boxes = tf.stack(boxes)
        # store search scales to facilitate target locating
        x_scale = tf.div(tf.to_float(x_sz), self.cfg.search_sz)
        search_scales = []
        for factor in self.scale_factors:
            search_scales.append(x_scale / factor)
        self.search_scales = tf.stack(search_scales)

        # multi-scale searching images
        avg_color = tf.reduce_mean(image, axis=(0, 1), name='avg_color')
        image_minus_avg = tf.expand_dims(image - avg_color, 0)
        search_images = tf.image.crop_and_resize(
            image_minus_avg, boxes,
            box_ind=tf.zeros((self.cfg.scale_num), tf.int32),
            crop_size=[self.cfg.search_sz, self.cfg.search_sz])
        search_images = search_images + avg_color

        # exemplar image
        begin = self.cfg.search_sz // 2 - self.cfg.exemplar_sz // 2
        exemplar_image = tf.expand_dims(
            search_images[self.cfg.scale_num // 2], axis=0)
        exemplar_image = tf.slice(
            exemplar_image, [0, begin, begin, 0],
            [-1, self.cfg.exemplar_sz, self.cfg.exemplar_sz, -1])

        # template embedding
        net = AlexNet()
        templates, _ = net(exemplar_image, trainable=self.cfg.trainable,
                           is_training=False)
        templates = tf.concat([
            templates for _ in range(self.cfg.scale_num)], axis=0)
        with tf.variable_scope('target_template'):
            with tf.variable_scope('State'):
                state = tf.get_variable(
                    'exemplar', initializer=tf.zeros(
                        templates.get_shape().as_list(),
                        dtype=templates.dtype),
                    trainable=False)
                with tf.control_dependencies([templates]):
                    self.init_op = tf.assign(state, templates,
                                             validate_shape=True)
                templates = state

        # response map
        search_embeds, _ = net(search_images, trainable=self.cfg.trainable,
                               is_training=False, reuse=True)
        with tf.variable_scope('detection'):
            def xcorr(x, z):
                x = tf.expand_dims(x, 0)
                z = tf.expand_dims(z, -1)
                return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1],
                                    padding='VALID', name='xcorr')

            output = tf.map_fn(
                lambda x: xcorr(x[0], x[1]),
                (search_embeds, templates),
                dtype=search_embeds.dtype)
            output = tf.squeeze(output, [1, 4])

            bias = tf.get_variable(
                'biases', [1], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                trainable=False)
            response = self.cfg.adjust_scale * output + bias

        # upsample response
        with tf.variable_scope('upsample'):
            response = tf.expand_dims(response, 3)
            response_sz = response.get_shape().as_list()[1:3]
            up_sz = [s * self.cfg.response_up for s in response_sz]
            response_up = tf.image.resize_images(
                response, up_sz, method=tf.image.ResizeMethod.BICUBIC,
                align_corners=True)
            self.response_up = tf.squeeze(response_up, [3])

    def setup_saver(self):
        ema = tf.train.ExponentialMovingAverage(0)
        variables = ema.variables_to_restore(moving_avg_variables=[])

        # filter out State variables
        variables = {k: v for k, v in variables.items()
                     if not 'State' in k}
        self.saver = tf.train.Saver(variables)

    def load_model(self, sess, checkpoint_path):
        self.saver.restore(sess, checkpoint_path)

    def init(self, sess, img_file, init_rect):
        out = sess.run(self.init_op, feed_dict={
            'filename:0': img_file,
            'box:0': init_rect})

        return out

    def detect(self, sess, img_file, last_rect):
        response_up, search_scales = sess.run(
            [self.response_up, self.search_scales], feed_dict={
            'filename:0': img_file,
            'box:0': last_rect})

        return response_up, search_scales


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamFC, self).__init__('SiamFC')
        self.parse_args(**kargs)

        # setup graph
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.model = GraphSiamFC()
        self.tf_graph.finalize()

        # setup Session
        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(graph=self.tf_graph, config=sess_config)

        # load checkpoint
        if net_path is not None:
            self.model.load_model(self.sess, net_path)

    def parse_args(self, **kargs):
        self.cfg = {
            'trainable': False,
            'exemplar_sz': 127,
            'search_sz': 255,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_up': 8,
            'total_stride': 8,
            'adjust_scale': 0.001}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

    def init(self, img_file, init_rect):
        self.box = init_rect
        self.model.init(self.sess, img_file, init_rect)

        # initialize parameters
        self.scale_factors = self.model.scale_factors
        self.response_sz = self.model.response_up.get_shape().as_list()[1]
        hann_1d = np.expand_dims(np.hanning(self.response_sz), axis=0)
        self.window = np.outer(hann_1d, hann_1d)
        self.window /= self.window.sum()
        self.scale_penalty = np.ones(self.cfg.scale_num) * self.cfg.scale_penalty
        self.scale_penalty[self.cfg.scale_num // 2] = 1.0
        self.initial_target_sz = self.box[2:]

    def update(self, img_file):
        response_up, search_scales = self.model.detect(
            self.sess, img_file, self.box)

        best_scale, best_loc = self._find_peak(response_up)
        self.box = self._locate(
            self.box, search_scales, best_scale, best_loc)

        return self.box

    def track(self, img_files, init_rect, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0, :] = init_rect
        speed_fps = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            start_time = time.time()
            if f == 0:
                self.init(img_file, init_rect)
            else:
                boxes[f, :] = self.update(img_file)
            elapsed_time = time.time() - start_time
            speed_fps[f] = 1. / elapsed_time

            if visualize:
                show_frame(Image.open(img_file), boxes[f], fig_n=1)

        return boxes, speed_fps

    def _find_peak(self, response):
        # find best scale
        max_responses = np.max(response, axis=(1, 2)) * self.scale_penalty
        best_scale = np.argmax(max_responses)

        # find peak location
        response = response[best_scale]
        response -= response.min()
        response /= (response.sum() + 1e-16)
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.window
        best_loc = np.unravel_index(response.argmax(), response.shape)

        return best_scale, np.array(best_loc, float)

    def _locate(self, box, search_scales, best_scale, best_loc):
        # update center
        disp_in_response = best_loc - (self.response_sz - 1) / 2
        disp_in_area = disp_in_response * self.cfg.total_stride / self.cfg.response_up
        disp_in_frame = disp_in_area / search_scales[best_scale]

        # 0-indexed
        center = (box[:2] - 1) + (box[2:] - 1) / 2
        center += disp_in_frame[::-1]

        # update scale
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[best_scale]
        target_scale = box[2:] * scale / self.initial_target_sz
        target_scale = np.clip(target_scale, 0.2, 5.0)
        target_sz = self.initial_target_sz * target_scale

        # convert to 1-indexed and left-top based
        box = np.concatenate([
            center - (target_sz - 1) / 2 + 1, target_sz])

        return box
