from __future__ import absolute_import, division

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from collections import namedtuple


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
            'bn_epsilon': 'kaiming_normal',
            'weight_decay': 5e-4}
        
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
            'variable_collections': {
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
            avtivation_fn=tf.nn.relu,
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
            'alexnet', 'alexnet', [inputs], reuse=reuse) as sc:
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
        # graph input: filename, bndbox
        # grapu output: init_op, response
        self.setup_graph()

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

    def setup_graph(self):
        # placeholders
        filename = tf.placeholder(tf.string, [], name='filename')
        bndbox = tf.placeholder(tf.float32, [4], name='bndbox')

        # convert bndbox to 0-indexed and center based [y, x, h, w]
        bndbox = tf.concat([
            bndbox[1] - (bndbox[3] - 1) / 2,
            bndbox[0] - (bndbox[2] - 1) / 2,
            bndbox[3], bndbox[2]], axis=0)
        center, target_sz = bndbox[:2]. bndbox[2:]

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
        scale_factors = self.cfg.scale_step ** scales
        boxes = []
        for factor in scale_factors:
            scaled_search_area = factor * x_sz
            image_sz_1 = tf.to_float(image_sz[:2] - 1)
            boxes.append(tf.concat([
                tf.div(center - (x_sz - 1) / 2, image_sz_1),
                tf.div(center + (x_sz - 1) / 2, image_sz_1)], axis=0))
        boxes = tf.stack(boxes)

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
            exemplar_image, [0, begin, begin],
            [1, self.cfg.exemplar_sz, self.cfg.exemplar_sz])

        # template embedding
        net = AlexNet()
        templates = net(exemplar_image, trainable=self.cfg.trainable,
                        is_training=False)
        templates = tf.stack([
            templates for _ in range(self.cfg.scale_num)])
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
        search_embeds = net(search_images, trainable=self.cfg.trainable,
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
                response, up_sz, method='bicubic', align_corners=True)
            self.response_up = tf.squeeze(response_up, [3])

    def init(self, sess, img_file, init_rect):
        out = sess.run(self.init_op, feed_dict={
            'filename:0': img_file,
            'bndbox:0': init_rect})
        
        return out

    def detect(self, sess, img_file, last_rect):
        response_up = sess.run(self.response_up, feed_dict={
            'filename:0': img_file,
            'bndbox:0': last_rect})
        
        return response_up


class TrackerSiamFC(Tracker):

    def __init__(self):
        super(TrackerSiamFC, self).__init__('SiamFC')

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.model = GraphSiamFC()
        self.tf_graph.finalize()
    
    def init(self, sess, img_file, init_rect):
        self.bndbox = init_rect
        self.model.init(sess, img_file, init_rect)
    
    def update(self, sess, img_file):
        response = self.model.infer(sess, img_file, self.bndbox)
        # TODO: locate target based on the response

    def track(self, img_files, init_rect, visualize=False):
        frame_num = len(img_files)
        bndboxes = np.zeros((frame_num, 4))
        bndboxes[0, :] = init_rect
        speed_fps = np.zeros(frame_num)

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(graph=self.tf_graph, config=sess_config) as sess:
            for f, img_file in enumerate(img_files):
                start_time = time.time()
                if f == 0:
                    self.init(image, init_rect)
                else:
                    bndboxes[f, :] = self.update(image)
                elapsed_time = time.time() - start_time
                speed_fps[f] = 1. / elapsed_time

                if visualize:
                    show_frame(Image.open(img_file), bndboxes[f], fig_n=1)

        return bndboxes, speed_fps
