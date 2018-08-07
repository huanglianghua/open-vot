from __future__ import absolute_import, division, print_function

import logging
import tensorflow as tf
import functools
import os
import os.path as osp
import numpy as np
import cv2
import time
from sacred import Experiment

from ..utils.misc_utils import get_center, get, auto_select_gpu, mkdir_p, sort_nicely, load_cfgs
from ..utils.infer_utils import get_exemplar_images, convert_bbox_format, Rectangle
from ..utils.viz import show_frame

slim = tf.contrib.slim
ex = Experiment()


def convolutional_alexnet_arg_scope(embed_config,
                                    trainable=True,
                                    is_training=False):
    """Defines the default arg scope.

    Args:
      embed_config: A dictionary which contains configurations for the embedding function.
      trainable: If the weights in the embedding function is trainable.
      is_training: If the embedding function is built for training.

    Returns:
      An `arg_scope` to use for the convolutional_alexnet models.
    """
    # Only consider the model to be in training mode if it's trainable.
    # This is vital for batch_norm since moving_mean and moving_variance
    # will get updated even if not trainable.
    is_model_training = trainable and is_training

    if get(embed_config, 'use_bn', True):
        batch_norm_scale = get(embed_config, 'bn_scale', True)
        batch_norm_decay = 1 - get(embed_config, 'bn_momentum', 3e-4)
        batch_norm_epsilon = get(embed_config, 'bn_epsilon', 1e-6)
        batch_norm_params = {
            "scale": batch_norm_scale,
            # Decay for the moving averages.
            "decay": batch_norm_decay,
            # Epsilon to prevent 0s in variance.
            "epsilon": batch_norm_epsilon,
            "trainable": trainable,
            "is_training": is_model_training,
            # Collection containing the moving mean and moving variance.
            "variables_collections": {
                "beta": None,
                "gamma": None,
                "moving_mean": ["moving_vars"],
                "moving_variance": ["moving_vars"],
            },
            'updates_collections': None,  # Ensure that updates are done within a frame
        }
        normalizer_fn = slim.batch_norm
    else:
        batch_norm_params = {}
        normalizer_fn = None

    weight_decay = get(embed_config, 'weight_decay', 5e-4)
    if trainable:
        weights_regularizer = slim.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None

    init_method = get(embed_config, 'init_method', 'kaiming_normal')
    if is_model_training:
        logging.info('embedding init method -- {}'.format(init_method))
    if init_method == 'kaiming_normal':
        # The same setting as siamese-fc
        initializer = slim.variance_scaling_initializer(
            factor=2.0, mode='FAN_OUT', uniform=False)
    else:
        initializer = slim.xavier_initializer()

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=weights_regularizer,
        weights_initializer=initializer,
        padding='VALID',
        trainable=trainable,
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.batch_norm], is_training=is_model_training) as arg_sc:
                return arg_sc


def convolutional_alexnet(inputs, reuse=None, scope='convolutional_alexnet'):
    """Defines the feature extractor of SiamFC.

    Args:
      inputs: a Tensor of shape [batch, h, w, c].
      reuse: if the weights in the embedding function are reused.
      scope: the variable scope of the computational graph.

    Returns:
      net: the computed features of the inputs.
      end_points: the intermediate outputs of the embedding function.
    """
    with tf.variable_scope(scope, 'convolutional_alexnet', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = inputs
            net = slim.conv2d(net, 96, [11, 11], 2, scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            with tf.variable_scope('conv2'):
                b1, b2 = tf.split(net, 2, 3)
                b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
                # The original implementation has bias terms for all convolution, but
                # it actually isn't necessary if the convolution layer is followed by a batch
                # normalization layer since batch norm will subtract the mean.
                b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
                net = tf.concat([b1, b2], 3)
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], 1, scope='conv3')
            with tf.variable_scope('conv4'):
                b1, b2 = tf.split(net, 2, 3)
                b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
                b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
                net = tf.concat([b1, b2], 3)
            # Conv 5 with only convolution, has bias
            with tf.variable_scope('conv5'):
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    b1, b2 = tf.split(net, 2, 3)
                    b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
                    b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
                net = tf.concat([b1, b2], 3)
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection)
            return net, end_points


convolutional_alexnet.stride = 8


class InferenceWrapper():
    """Model wrapper class for performing inference with a siamese model."""

    def __init__(self):
        self.image = None
        self.target_bbox_feed = None
        self.search_images = None
        self.embeds = None
        self.templates = None
        self.init = None
        self.model_config = None
        self.track_config = None
        self.response_up = None

    def build_graph_from_config(self, model_config, track_config, checkpoint_path):
        """Build the inference graph and return a restore function."""
        self.build_model(model_config, track_config)
        ema = tf.train.ExponentialMovingAverage(0)
        variables_to_restore = ema.variables_to_restore(
            moving_avg_variables=[])

        # Filter out State variables
        variables_to_restore_filterd = {}
        for key, value in variables_to_restore.items():
            if key.split('/')[1] != 'State':
                variables_to_restore_filterd[key] = value

        saver = tf.train.Saver(variables_to_restore_filterd)

        if osp.isdir(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if not checkpoint_path:
                raise ValueError(
                    "No checkpoint file found in: {}".format(checkpoint_path))

        def _restore_fn(sess):
            logging.info("Loading model from checkpoint: %s", checkpoint_path)
            saver.restore(sess, checkpoint_path)
            logging.info("Successfully loaded checkpoint: %s",
                         os.path.basename(checkpoint_path))

        return _restore_fn

    def build_model(self, model_config, track_config):
        self.model_config = model_config
        self.track_config = track_config

        self.build_inputs()
        self.build_search_images()
        self.build_template()
        self.build_detection()
        self.build_upsample()
        self.dumb_op = tf.no_op('dumb_operation')

    def build_inputs(self):
        filename = tf.placeholder(tf.string, [], name='filename')
        image_file = tf.read_file(filename)
        image = tf.image.decode_jpeg(
            image_file, channels=3, dct_method="INTEGER_ACCURATE")
        image = tf.to_float(image)
        self.image = image
        self.target_bbox_feed = tf.placeholder(dtype=tf.float32,
                                               shape=[4],
                                               name='target_bbox_feed')  # center's y, x, height, width

    def build_search_images(self):
        """Crop search images from the input image based on the last target position

        1. The input image is scaled such that the area of target&context takes up to (scale_factor * z_image_size) ^ 2
        2. Crop an image patch as large as x_image_size centered at the target center.
        3. If the cropped image region is beyond the boundary of the input image, mean values are padded.
        """
        model_config = self.model_config
        track_config = self.track_config

        size_z = model_config['z_image_size']
        size_x = track_config['x_image_size']
        context_amount = 0.5

        num_scales = track_config['num_scales']
        scales = np.arange(num_scales) - get_center(num_scales)
        assert np.sum(scales) == 0, 'scales should be symmetric'
        search_factors = [track_config['scale_step'] ** x for x in scales]

        frame_sz = tf.shape(self.image)
        target_yx = self.target_bbox_feed[0:2]
        target_size = self.target_bbox_feed[2:4]
        avg_chan = tf.reduce_mean(self.image, axis=(0, 1), name='avg_chan')

        # Compute base values
        base_z_size = target_size
        base_z_context_size = base_z_size + \
            context_amount * tf.reduce_sum(base_z_size)
        base_s_z = tf.sqrt(tf.reduce_prod(
            base_z_context_size))  # Canonical size
        base_scale_z = tf.div(tf.to_float(size_z), base_s_z)
        d_search = (size_x - size_z) / 2.0
        base_pad = tf.div(d_search, base_scale_z)
        base_s_x = base_s_z + 2 * base_pad
        base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

        boxes = []
        for factor in search_factors:
            s_x = factor * base_s_x
            frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
            topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
            bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
            box = tf.concat([topleft, bottomright], axis=0)
            boxes.append(box)
        boxes = tf.stack(boxes)

        scale_xs = []
        for factor in search_factors:
            scale_x = base_scale_x / factor
            scale_xs.append(scale_x)
        self.scale_xs = tf.stack(scale_xs)

        # Note we use different padding values for each image
        # while the original implementation uses only the average value
        # of the first image for all images.
        image_minus_avg = tf.expand_dims(self.image - avg_chan, 0)
        image_cropped = tf.image.crop_and_resize(image_minus_avg, boxes,
                                                 box_ind=tf.zeros(
                                                     (track_config['num_scales']), tf.int32),
                                                 crop_size=[size_x, size_x])
        self.search_images = image_cropped + avg_chan

    def get_image_embedding(self, images, reuse=None):
        config = self.model_config['embed_config']
        arg_scope = convolutional_alexnet_arg_scope(config,
                                                    trainable=config['train_embedding'],
                                                    is_training=False)

        @functools.wraps(convolutional_alexnet)
        def embedding_fn(images, reuse=False):
            with slim.arg_scope(arg_scope):
                return convolutional_alexnet(images, reuse=reuse)

        embed, _ = embedding_fn(images, reuse)

        return embed

    def build_template(self):
        model_config = self.model_config
        track_config = self.track_config

        # Exemplar image lies at the center of the search image in the first frame
        exemplar_images = get_exemplar_images(self.search_images, [model_config['z_image_size'],
                                                                   model_config['z_image_size']])
        templates = self.get_image_embedding(exemplar_images)
        center_scale = int(get_center(track_config['num_scales']))
        center_template = tf.identity(templates[center_scale])
        templates = tf.stack(
            [center_template for _ in range(track_config['num_scales'])])

        with tf.variable_scope('target_template'):
            # Store template in Variable such that we don't have to feed this template every time.
            with tf.variable_scope('State'):
                state = tf.get_variable('exemplar',
                                        initializer=tf.zeros(
                                            templates.get_shape().as_list(), dtype=templates.dtype),
                                        trainable=False)
                with tf.control_dependencies([templates]):
                    self.init = tf.assign(
                        state, templates, validate_shape=True)
                self.templates = state

    def build_detection(self):
        self.embeds = self.get_image_embedding(self.search_images, reuse=True)
        with tf.variable_scope('detection'):
            def _translation_match(x, z):
                # [batch, in_height, in_width, in_channels]
                x = tf.expand_dims(x, 0)
                # [filter_height, filter_width, in_channels, out_channels]
                z = tf.expand_dims(z, -1)
                return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

            output = tf.map_fn(
                lambda x: _translation_match(x[0], x[1]),
                (self.embeds, self.templates), dtype=self.embeds.dtype)  # of shape [16, 1, 17, 17, 1]
            output = tf.squeeze(output, [1, 4])  # of shape e.g. [16, 17, 17]

            bias = tf.get_variable('biases', [1],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(
                                       0.0, dtype=tf.float32),
                                   trainable=False)
            response = self.model_config['adjust_response_config']['scale'] * output + bias
            self.response = response

    def build_upsample(self):
        """Upsample response to obtain finer target position"""
        with tf.variable_scope('upsample'):
            response = tf.expand_dims(self.response, 3)
            up_method = self.track_config['upsample_method']
            methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                       'bicubic': tf.image.ResizeMethod.BICUBIC}
            up_method = methods[up_method]
            response_spatial_size = self.response.get_shape().as_list()[1:3]
            up_size = [s * self.track_config['upsample_factor']
                       for s in response_spatial_size]
            response_up = tf.image.resize_images(response,
                                                 up_size,
                                                 method=up_method,
                                                 align_corners=True)
            response_up = tf.squeeze(response_up, [3])
            self.response_up = response_up

    def initialize(self, sess, input_feed):
        image_path, target_bbox = input_feed
        scale_xs, _ = sess.run([self.scale_xs, self.init],
                               feed_dict={'filename:0': image_path,
                                          "target_bbox_feed:0": target_bbox, })
        return scale_xs

    def inference_step(self, sess, input_feed):
        image_path, target_bbox = input_feed
        log_level = self.track_config['log_level']
        image_cropped_op = self.search_images if log_level > 0 else self.dumb_op
        image_cropped, scale_xs, response_output = sess.run(
            fetches=[image_cropped_op, self.scale_xs, self.response_up],
            feed_dict={
                "filename:0": image_path,
                "target_bbox_feed:0": target_bbox, })

        output = {
            'image_cropped': image_cropped,
            'scale_xs': scale_xs,
            'response': response_output}
        return output, None


class TargetState(object):
    """Represent the target state."""

    def __init__(self, bbox, search_pos, scale_idx):
        self.bbox = bbox  # (cx, cy, w, h) in the original image
        self.search_pos = search_pos  # target center position in the search image
        self.scale_idx = scale_idx  # scale index in the searched scales


class TrackerSiamFC(object):
    """Tracker based on the siamese model."""

    def __init__(self):
        self.name = 'SiamFC'
        checkpoint = 'Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained'
        os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
        model_config, _, track_config = load_cfgs(checkpoint)
        track_config['log_level'] = 1

        g = tf.Graph()
        with g.as_default():
            model = InferenceWrapper()
            self.restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint)
        g.finalize()
        self.g = g

        if not osp.isdir(track_config['log_dir']):
            logging.info('Creating inference directory: %s', track_config['log_dir'])
            mkdir_p(track_config['log_dir'])

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess_config = tf.ConfigProto(gpu_options=gpu_options)

        self.siamese_model = model
        self.model_config = model_config
        self.track_config = track_config

        self.num_scales = track_config['num_scales']
        logging.info('track num scales -- {}'.format(self.num_scales))
        scales = np.arange(self.num_scales) - get_center(self.num_scales)
        self.search_factors = [
            self.track_config['scale_step'] ** x for x in scales]

        self.x_image_size = track_config['x_image_size']  # Search image size
        self.window = None  # Cosine window
        self.log_level = track_config['log_level']

    def init(self, sess, filename, first_bbox):
        # Get initial target bounding box and convert to center based
        bbox = convert_bbox_format(first_bbox, 'center-based')

        # Feed in the first frame image to set initial state.
        bbox_feed = [bbox.y, bbox.x, bbox.height, bbox.width]
        input_feed = [filename, bbox_feed]
        frame2crop_scale = self.siamese_model.initialize(sess, input_feed)

        # Storing target state
        self.original_target_height = bbox.height
        self.original_target_width = bbox.width
        self.search_center = np.array([get_center(self.x_image_size),
                                       get_center(self.x_image_size)])
        self.current_target_state = TargetState(bbox=bbox,
                                                search_pos=self.search_center,
                                                scale_idx=int(get_center(self.num_scales)))

        include_first = get(self.track_config, 'include_first', False)
        logging.info('Tracking include first -- {}'.format(include_first))

    def update(self, sess, filename):
        bbox_feed = [self.current_target_state.bbox.y, self.current_target_state.bbox.x,
                     self.current_target_state.bbox.height, self.current_target_state.bbox.width]
        input_feed = [filename, bbox_feed]

        outputs, metadata = self.siamese_model.inference_step(sess, input_feed)
        search_scale_list = outputs['scale_xs']
        response = outputs['response']
        response_size = response.shape[1]

        # Choose the scale whole response map has the highest peak
        if self.num_scales > 1:
            response_max = np.max(response, axis=(1, 2))
            penalties = self.track_config['scale_penalty'] * \
                np.ones((self.num_scales))
            current_scale_idx = int(get_center(self.num_scales))
            penalties[current_scale_idx] = 1.0
            response_penalized = response_max * penalties
            best_scale = np.argmax(response_penalized)
        else:
            best_scale = 0

        response = response[best_scale]

        with np.errstate(all='raise'):  # Raise error if something goes wrong
            response = response - np.min(response)
            response = response / np.sum(response)

        if self.window is None:
            window = np.dot(np.expand_dims(np.hanning(response_size), 1),
                            np.expand_dims(np.hanning(response_size), 0))
            self.window = window / np.sum(window)  # normalize window
        window_influence = self.track_config['window_influence']
        response = (1 - window_influence) * response + \
            window_influence * self.window

        # Find maximum response
        r_max, c_max = np.unravel_index(response.argmax(),
                                        response.shape)

        # Convert from crop-relative coordinates to frame coordinates
        p_coor = np.array([r_max, c_max])
        # displacement from the center in instance final representation ...
        disp_instance_final = p_coor - get_center(response_size)
        # ... in instance feature space ...
        upsample_factor = self.track_config['upsample_factor']
        disp_instance_feat = disp_instance_final / upsample_factor
        # ... Avoid empty position ...
        r_radius = int(response_size / upsample_factor / 2)
        disp_instance_feat = np.maximum(np.minimum(
            disp_instance_feat, r_radius), -r_radius)
        # ... in instance input ...
        disp_instance_input = disp_instance_feat * \
            self.model_config['embed_config']['stride']
        # ... in instance original crop (in frame coordinates)
        disp_instance_frame = disp_instance_input / \
            search_scale_list[best_scale]
        # Position within frame in frame coordinates
        y = self.current_target_state.bbox.y
        x = self.current_target_state.bbox.x
        y += disp_instance_frame[0]
        x += disp_instance_frame[1]

        # Target scale damping and saturation
        target_scale = self.current_target_state.bbox.height / self.original_target_height
        search_factor = self.search_factors[best_scale]
        # damping factor for scale update
        scale_damp = self.track_config['scale_damp']
        target_scale *= ((1 - scale_damp) * 1.0 + scale_damp * search_factor)
        target_scale = np.maximum(0.2, np.minimum(5.0, target_scale))

        # Some book keeping
        height = self.original_target_height * target_scale
        width = self.original_target_width * target_scale
        self.current_target_state.bbox = Rectangle(x, y, width, height)
        self.current_target_state.scale_idx = best_scale
        self.current_target_state.search_pos = self.search_center + disp_instance_input

        assert 0 <= self.current_target_state.search_pos[0] < self.x_image_size, \
            'target position in feature space should be no larger than input image size'
        assert 0 <= self.current_target_state.search_pos[1] < self.x_image_size, \
            'target position in feature space should be no larger than input image size'

        reported_bbox = convert_bbox_format(
            self.current_target_state.bbox, 'top-left-based')
        out_bbox = np.asarray([reported_bbox.x, reported_bbox.y,
                               reported_bbox.width, reported_bbox.height])

        return out_bbox

    def track(self, img_files, init_rect, visualize=False):
        with tf.Session(graph=self.g, config=self.sess_config) as sess:
            self.restore_fn(sess)
            
            bb = init_rect
            init_bb = Rectangle(bb[0] - 1, bb[1] - 1,
                                bb[2], bb[3])  # 0-index in python

            frame_num = len(img_files)
            bndboxes = np.zeros((frame_num, 4))
            bndboxes[0, :] = init_rect
            speed_fps = np.zeros(frame_num)

            for f, filename in enumerate(img_files):
                start_time = time.time()
                if f == 0:
                    self.init(sess, filename, init_bb)
                else:
                    bndboxes[f, :] = self.update(sess, filename)
                elapsed_time = time.time() - start_time
                speed_fps[f] = 1. / elapsed_time

                if visualize:
                    show_frame(cv2.imread(filename)[:, :, ::-1],
                               bndboxes[f, :], fig_n=1)

        return bndboxes, speed_fps

    def _track(self, sess, first_bbox, frames, logdir='/tmp'):
        self.init(sess, frames[0], first_bbox)

        for i, filename in enumerate(frames):
            bbox = self.update(sess, filename)

            frame_image = cv2.imread(filename)
            frame_image = cv2.rectangle(
                frame_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[0] + bbox[2]),
                 int(bbox[1] + bbox[3])),
                (0, 0, 255), 3)
            cv2.imshow('window', frame_image)
            cv2.waitKey(1)
