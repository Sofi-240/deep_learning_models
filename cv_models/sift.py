from typing import Union
import tensorflow as tf
from cv_models.utils import PI, gaussian_kernel, compute_extrema3D, make_neighborhood3D, \
    make_neighborhood2D, gaussian_blur, compute_central_gradient3D, compute_central_gradient2D, compute_hessian_3D
from tensorflow.python.keras import backend
from tensorflow import keras
from keras.layers import RNN
from collections import namedtuple

backend.set_floatx('float32')
linalg_ops = tf.linalg
math_ops = tf.math
bitwise_ops = tf.bitwise


class OctaveRNN(keras.layers.Layer):
    def __init__(self, shape, gaussian_kernels, name=None):
        super(OctaveRNN, self).__init__(name=name or 'OctaveRNN')
        b_, h_, w_, d_ = shape
        self.state_size = tf.TensorShape([*shape])
        self.output_size = [tf.TensorShape([*shape]), tf.TensorShape([*shape]),
                            tf.TensorShape([b_, h_ - 2, w_ - 2, d_, 2])]
        self.gaussian_kernels = [
            tf.Variable(kernel, name=f'kernel{i}', trainable=False) for i, kernel in
            enumerate(gaussian_kernels)
        ]
        kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
        kx = tf.reshape(kx, shape=(3, 3, 1, 1))
        ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
        ky = tf.reshape(ky, shape=(3, 3, 1, 1))
        self.kernels_grad = tf.Variable(tf.concat((kx, ky), axis=-1), name='kernels_grad', trainable=False)

    def call(self, inputs, states, **kwargs):
        def _conv(X, H):
            k_ = int(H.get_shape()[0])
            paddings = tf.constant(
                [[0, 0], [k_ // 2, k_ // 2], [k_ // 2, k_ // 2], [0, 0]], dtype=tf.int32
            )
            X_pad = tf.pad(X, paddings, mode='SYMMETRIC')
            H = tf.reshape(H, shape=(k_, k_, 1, 1))
            return tf.nn.convolution(X_pad, H, padding='VALID')

        _iter = tf.cast(tf.reshape(inputs, ()), tf.int32)

        if _iter == 0:
            h = states[0]
            dog = h * 0.0
        else:
            h = _conv(tf.squeeze(states[0], axis=0), self.gaussian_kernels[_iter - 1])
            h = tf.expand_dims(h, axis=0)
            dog = math_ops.subtract(h, states[0])

        gradient = tf.nn.convolution(tf.squeeze(h, axis=0), self.kernels_grad, padding='VALID')
        gradient = tf.expand_dims(gradient, axis=0)

        return [h, dog, gradient], [h]


class SIFT:

    def __init__(self,
                 sigma: float = 1.6, assume_blur_sigma: float = 0.5, n_intervals: int = 3,
                 n_octaves: Union[int, None] = 4, n_iterations: int = 5, with_descriptors: bool = True,
                 name: Union[str, None] = None):
        self.name = name or 'SIFT'
        self.__DTYPE = backend.floatx()
        self.epsilon = 1e-05
        self.with_descriptors = with_descriptors
        self.n_octaves = None if n_octaves is None else tf.cast(n_octaves, dtype=tf.int32)
        self.n_intervals = tf.cast(n_intervals, dtype=tf.int32)
        self.n_iterations = tf.cast(n_iterations, dtype=tf.int32)
        self.sigma = tf.cast(sigma, dtype=self.__DTYPE)
        self.assume_blur_sigma = tf.cast(assume_blur_sigma, dtype=self.__DTYPE)
        self.key_points = None
        self.descriptors_vectors = None
        self.octave_pyramid = None
        graph_args = namedtuple(
            'SIFT_args',
            'n_bins eigen_ration peak_ratio contrast_threshold scale_factor extrema_offset radius_factor con border_width'
        )
        self.graph_args = graph_args(36, 10, 0.8, 0.04, 1.5, 0.5, 3, 3, (3, 3, 0))
        descriptors_args = namedtuple('descriptors_args', 'n_bins window_width scale_multiplier descriptor_max_value')
        self.descriptors_args = descriptors_args(8, 4, 3, 0.2)
        self.__inputs_shape = None
        self.__octaves_pack = namedtuple('octaves_pack', 'gaussian, dx, dy, magnitude, orientation')
        self.__key_points_pack = namedtuple('key_points_pack', 'pt, size, angle, octave, octave_id, response')

    def compute_default_N_octaves(self, image_shape: Union[tf.Tensor, list, tuple], min_shape: int = 0) -> tf.Tensor:
        assert len(image_shape) == 4
        b, h, w, d = tf.unstack(image_shape, num=4)

        s_ = tf.cast(min([h, w]), dtype=self.__DTYPE)
        diff = math_ops.log(s_)
        if min_shape > 1:
            diff = diff - math_ops.log(tf.cast(min_shape, dtype=self.__DTYPE))

        n_octaves = tf.round(diff / math_ops.log(2.0)) + 1
        return tf.cast(n_octaves, tf.int32)

    def __valid_input(self, inputs: tf.Tensor) -> None:
        _shape = inputs.get_shape()
        _ndims = len(_shape)
        if _ndims != 4 or _shape[-1] != 1:
            raise ValueError(
                'expected the inputs to be grayscale images with size of (None, h, w, 1)'
            )
        self.__inputs_shape = _shape

    def __localize_extrema(self, middle_pixel_cords: tf.Tensor, dOg_array: tf.Tensor,
                           octave_index: Union[int, float]) -> tuple[Union[tf.Tensor, None], Union[tf.Tensor, None]]:

        dog_shape_ = dOg_array.shape
        args = self.graph_args
        cube_neighbor = make_neighborhood3D(middle_pixel_cords, con=args.con, origin_shape=dog_shape_)
        if cube_neighbor.shape[0] == 0:
            return None, None

        # the size can change because the make_neighborhood3D return only the valid indexes
        mid_index = (args.con ** 3) // 2
        _, middle_pixel_cords, _ = tf.split(cube_neighbor, [mid_index, 1, mid_index], axis=1)
        middle_pixel_cords = tf.reshape(middle_pixel_cords, shape=(-1, 4))

        cube_values = tf.gather_nd(dOg_array, tf.reshape(cube_neighbor, shape=(-1, 4))) / 255.0
        cube_values = tf.reshape(cube_values, (-1, args.con, args.con, args.con))

        cube_len_ = cube_values.get_shape()[0]

        mid_cube_values = tf.slice(cube_values, [0, 1, 1, 1], [cube_len_, 1, 1, 1])
        mid_cube_values = tf.reshape(mid_cube_values, shape=(-1,))

        grad = compute_central_gradient3D(cube_values)
        grad = tf.reshape(grad, shape=(-1, 3, 1))

        hess = compute_hessian_3D(cube_values)
        hess = tf.reshape(hess, shape=(-1, 3, 3))

        extrema_update = - linalg_ops.lstsq(hess, grad, l2_regularizer=0.0, fast=False)
        extrema_update = tf.squeeze(extrema_update, axis=-1)

        next_step_cords, kp_cond_1 = self.__split_extrema_cond(extrema_update, middle_pixel_cords)

        dot_ = tf.reduce_sum(
            tf.multiply(
                extrema_update[:, tf.newaxis, tf.newaxis, ...],
                tf.transpose(grad, perm=(0, 2, 1))[:, tf.newaxis, ...]
            ), axis=-1, keepdims=False
        )
        dot_ = tf.reshape(dot_, shape=(-1,))
        update_response = mid_cube_values + 0.5 * dot_

        kp_cond_2 = math_ops.greater_equal(
            math_ops.abs(update_response) * int(self.n_intervals), args.contrast_threshold - self.epsilon
        )

        hess_xy = tf.slice(hess, [0, 0, 0], [cube_len_, 2, 2])
        hess_xy_trace = linalg_ops.trace(hess_xy)
        hess_xy_det = linalg_ops.det(hess_xy)

        kp_cond_3 = math_ops.logical_and(
            math_ops.greater(hess_xy_det, 0.0),
            math_ops.less(args.eigen_ration * (hess_xy_trace ** 2), ((args.eigen_ration + 1) ** 2) * hess_xy_det)
        )
        sure_key_points = math_ops.logical_and(math_ops.logical_and(kp_cond_1, kp_cond_2), kp_cond_3)

        # -------------------------------------------------------------------------
        kp_extrema_update = tf.boolean_mask(extrema_update, sure_key_points)

        if kp_extrema_update.shape[0] == 0:
            return next_step_cords, None

        kp_cords = tf.cast(tf.boolean_mask(middle_pixel_cords, sure_key_points), dtype=self.__DTYPE)
        octave_index = tf.cast(octave_index, dtype=self.__DTYPE)

        ex, ey, ez = tf.unstack(kp_extrema_update, num=3, axis=1)
        cd, cy, cx, cz = tf.unstack(kp_cords, num=4, axis=1)

        kp_pt = tf.stack(
            (cd, (cy + ey) * (2 ** octave_index), (cx + ex) * (2 ** octave_index), cz), axis=-1
        )

        kp_octave = octave_index + cz * (2 ** 8) + tf.round((ez + 0.5) * 255.0) * (2 ** 16)

        kp_size = self.sigma * (2 ** ((cz + ez) / tf.cast(self.n_intervals, dtype=self.__DTYPE))) * (
                2 ** (octave_index + 1.0))
        kp_response = math_ops.abs(tf.boolean_mask(update_response, sure_key_points))

        octave_index = tf.ones_like(kp_size) * octave_index
        angle = tf.ones_like(kp_size) * -1.0
        key_point_wrap = [
            tf.reshape(kp_pt, (-1, 4)),
            tf.reshape(kp_size, (-1, 1)),
            tf.reshape(angle, (-1, 1)),
            tf.reshape(kp_octave, (-1, 1)),
            tf.reshape(octave_index, (-1, 1)),
            tf.reshape(kp_response, (-1, 1))

        ]
        key_point_wrap = tf.concat(key_point_wrap, axis=-1)
        return next_step_cords, key_point_wrap

    def __split_extrema_cond(self, extrema: tf.Tensor, current_cords: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        shape_ = extrema.get_shape()
        assert shape_[-1] == 3
        assert len(shape_) == 2

        cond = math_ops.less(math_ops.reduce_max(math_ops.abs(extrema), axis=-1), self.graph_args.extrema_offset)
        next_step_cords_temp = tf.boolean_mask(current_cords, ~cond)

        if next_step_cords_temp.shape[0] == 0:
            return None, cond

        extrema_shift = tf.boolean_mask(extrema, ~cond)

        ex, ey, ez = tf.unstack(extrema_shift, num=3, axis=1)
        positions_shift = tf.pad(
            tf.stack((ey, ex, ez), axis=-1),
            paddings=tf.constant([[0, 0], [1, 0]], dtype=tf.int32),
            constant_values=0.0
        )
        next_step_cords = next_step_cords_temp + tf.cast(tf.round(positions_shift), dtype=tf.int64)
        return next_step_cords, cond

    def __compute_histogram(self, key_points: tuple, octaves: list) -> tf.Tensor:
        args = self.graph_args
        key_points = self.__key_points_pack(*key_points)
        octaves = [self.__octaves_pack(*oc) for oc in octaves]

        histogram = tf.zeros(shape=(key_points.pt.shape[0], args.n_bins), dtype=tf.float32)

        scale = args.scale_factor * key_points.size / (2 ** key_points.octave_id + 1)

        radius = tf.cast(tf.round(args.radius_factor * scale), dtype=tf.int32)
        weight_factor = -0.5 / (scale ** 2)

        _prob = 1.0 / (2 ** key_points.octave_id)
        _one = tf.ones_like(_prob)
        _prob = tf.stack((_one, _prob, _prob, _one), axis=-1)
        _prob = tf.squeeze(_prob)

        region_center = tf.cast(key_points.pt * _prob, dtype=tf.int64) - tf.constant([[0, 1, 1, 0]], dtype=tf.int64)

        con = tf.reshape((radius * 2) + 1, (-1,))
        octave_id = tf.reshape(tf.cast(key_points.octave_id, tf.int32), (-1,))

        oct_shapes = [tf.reshape(tf.shape(a.dx), (1, 4)) for a in octaves]
        oct_shapes = tf.gather(tf.concat(oct_shapes, axis=0), octave_id)

        maximum_con = self.compute_max_con(region_center, oct_shapes)
        con = math_ops.minimum(con, maximum_con)
        parallel = self.compute_parallel_jobs(con, octave_id)

        for curr_parallel in parallel:
            curr_octave = octaves[curr_parallel.octave_id]
            curr_neighbor = make_neighborhood2D(tf.constant([[0, 0, 0, 0]], dtype=tf.int64), con=curr_parallel.con)

            curr_region = tf.gather(region_center, curr_parallel.idx)
            curr_block = tf.expand_dims(curr_region, axis=1) + curr_neighbor

            curr_magnitude = tf.gather_nd(curr_octave.magnitude, tf.reshape(curr_block, (-1, 4)))
            curr_orientation = tf.gather_nd(curr_octave.orientation, tf.reshape(curr_block, (-1, 4)))

            curr_neighbor = tf.cast(curr_neighbor, dtype=tf.float32)
            _, curr_y, curr_x, _ = tf.split(curr_neighbor, [1, 1, 1, 1], axis=-1)

            curr_weight = tf.gather(weight_factor, curr_parallel.idx)
            curr_weight = math_ops.exp(curr_weight * tf.squeeze((curr_y ** 2 + curr_x ** 2), axis=-1))

            curr_hist_index = tf.cast(tf.round(curr_orientation * args.n_bins / 360.), dtype=tf.int64)
            curr_hist_index = curr_hist_index % args.n_bins

            curr_hist_id = tf.ones_like(curr_weight, dtype=tf.int64) * tf.reshape(curr_parallel.idx, (-1, 1))
            curr_weight = tf.reshape(curr_weight, (-1,))

            curr_hist_index = tf.concat((tf.reshape(curr_hist_id, (-1, 1)), tf.reshape(curr_hist_index, (-1, 1))),
                                        axis=-1)

            histogram = tf.tensor_scatter_nd_add(histogram, curr_hist_index, curr_weight * curr_magnitude)

        return histogram

    def call(self, inputs: tf.Tensor):
        def unpack_rnn(x, dim):
            if dim == 4: x = tf.squeeze(x, axis=-1)
            x = tf.squeeze(x, axis=0)
            x = tf.transpose(x, perm=[*range(1, dim), 0])
            return x

        def down_sample(x, size):
            _h, _w, _d = size
            _, x, _ = tf.split(x, [_d - 3, 1, _d - (_d - 3) - 1], axis=-1)
            x = tf.image.resize(x, size=[_h, _w], method='nearest')
            return x

        def body(cords, x, oc_id, static_keys):
            cords, keys = self.__localize_extrema(cords, x, oc_id)
            if cords is None:
                cords = tf.constant([[]], shape=(0, 4), dtype=tf.int64)
            if keys is None:
                return [cords, x, oc_id, static_keys]
            static_keys = tf.concat((static_keys, keys), axis=0)
            return [cords, x, oc_id, static_keys]

        def cond_check(cords, x, oc_id, static_keys):
            return cords.get_shape()[0] != 0

        self.__valid_input(inputs)
        B_, H_, W_, D_ = self.__inputs_shape
        args = self.graph_args

        X_base = tf.image.resize(inputs, size=[H_ * 2, W_ * 2], method='bilinear', name='X_base')

        delta_sigma = (self.sigma ** 2) - ((2 * self.assume_blur_sigma) ** 2)
        delta_sigma = math_ops.sqrt(tf.maximum(delta_sigma, 0.64))

        X_base = gaussian_blur(X_base, kernel_size=0, sigma=delta_sigma)

        # generate the gaussian kernels
        images_per_octaves = self.n_intervals + 3
        K = 2 ** (1 / self.n_intervals)
        K = tf.cast(K, dtype=self.__DTYPE)

        sigma_prev = (K ** tf.cast(tf.range(images_per_octaves - 1), dtype=self.__DTYPE)) * self.sigma
        sigmas = math_ops.sqrt((K * sigma_prev) ** 2 - sigma_prev ** 2)
        sigmas = math_ops.maximum(sigmas, 0.8)

        gaussian_kernels = [gaussian_kernel(kernel_size=0, sigma=s) for s in sigmas]

        min_shape = int(tf.shape(gaussian_kernels[-1])[0])
        max_n_octaves = self.compute_default_N_octaves(image_shape=tf.shape(X_base), min_shape=min_shape)
        if self.n_octaves is not None and max_n_octaves > self.n_octaves:
            max_n_octaves = tf.cast(self.n_octaves, dtype=tf.int32)
        self.n_octaves = max_n_octaves

        n_iterations = int(self.n_iterations)
        threshold = tf.floor(0.5 * args.contrast_threshold / 3.0 * 255.0)

        _, hi, wi, _ = X_base.get_shape()
        _N_per_oc = len(gaussian_kernels) + 1
        iterator = tf.reshape(tf.cast(tf.range(_N_per_oc), tf.float32), (1, -1, 1))

        key_points = tf.constant([[]], shape=(0, 9), dtype=tf.float32)
        self.octave_pyramid = []

        for oc in tf.range(self.n_octaves):
            lay = OctaveRNN((B_, hi, wi, D_), gaussian_kernels, name=f'{oc}RNN')
            lay = RNN(lay, return_sequences=True)
            gauss, DoG, gradient = lay(iterator, initial_state=tf.expand_dims(X_base, axis=0))

            gauss = unpack_rnn(gauss, 4)

            DoG = unpack_rnn(DoG, 4)
            _, DoG = tf.split(DoG, [1, _N_per_oc - 1], axis=-1)

            continue_search = compute_extrema3D(
                DoG, threshold=threshold, con=args.con, border_width=args.border_width, epsilon=self.epsilon
            )
            if continue_search.shape[0] != 0:
                params = tf.while_loop(
                    cond=cond_check,
                    body=body,
                    loop_vars=[continue_search, DoG, oc, key_points],
                    maximum_iterations=n_iterations
                )
                key_points = params[-1]
            gradient = unpack_rnn(gradient, 5)
            dx, dy = tf.unstack(gradient, 2, axis=3)
            magnitude = math_ops.sqrt(dx * dx + dy * dy)
            orientation = math_ops.atan2(dx, dy) * (180.0 / PI)
            self.octave_pyramid.append(self.__octaves_pack(gauss, dx, dy, magnitude, orientation))
            hi = hi // 2
            wi = wi // 2
            X_base = down_sample(gauss, (hi, wi, _N_per_oc))

        key_points = self.__key_points_pack(*tf.split(key_points, [4, 1, 1, 1, 1, 1], axis=-1))
        histogram = self.__compute_histogram(key_points, self.octave_pyramid)

        gaussian1D = tf.constant([1, 4, 6, 4, 1], dtype=tf.float32) / 16.0
        gaussian1D = tf.reshape(gaussian1D, shape=(-1, 1, 1))

        histogram_pad = tf.pad(
            tf.expand_dims(histogram, axis=-1), paddings=tf.constant([[0, 0], [2, 2], [0, 0]], dtype=tf.int32),
            constant_values=0.0
        )

        smooth_histogram = tf.nn.convolution(histogram_pad, gaussian1D, padding='VALID')
        smooth_histogram = tf.squeeze(smooth_histogram, axis=-1)

        orientation_max = tf.reduce_max(smooth_histogram, axis=-1)
        value_cond = tf.repeat(tf.reshape(orientation_max, shape=(-1, 1)), repeats=args.n_bins, axis=-1)
        value_cond = value_cond * args.peak_ratio

        cond = math_ops.logical_and(
            math_ops.greater(smooth_histogram, tf.roll(smooth_histogram, shift=1, axis=1)),
            math_ops.greater(smooth_histogram, tf.roll(smooth_histogram, shift=-1, axis=1))
        )
        cond = math_ops.logical_and(
            cond, math_ops.greater_equal(smooth_histogram, value_cond)
        )

        peak_index = tf.where(cond)
        peak_value = tf.boolean_mask(smooth_histogram, cond)

        p_id, p_idx = tf.unstack(peak_index, num=2, axis=-1)

        left_index = (p_idx - 1) % args.n_bins
        right_index = (p_idx + 1) % args.n_bins

        left_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, left_index), axis=-1))
        right_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, right_index), axis=-1))

        interpolated_peak_index = ((tf.cast(p_idx, dtype=tf.float32) + 0.5 * (left_value - right_value)) / (
                left_value - (2 * peak_value) + right_value)) % args.n_bins

        orientation = 360. - interpolated_peak_index * 360. / args.n_bins

        orientation = tf.where(math_ops.less(math_ops.abs(orientation), 1e-7), 0.0, orientation)

        p_id = tf.reshape(p_id, (-1,))
        new_kp = {}

        for key, itm in key_points._asdict().items():
            if key == 'angle':
                new_kp[key] = tf.reshape(orientation, (-1, 1))
                continue
            n = 1
            if key == 'pt': n = 4
            new_kp[key] = tf.reshape(tf.gather(itm, p_id), (-1, n))

        self.key_points = self.__key_points_pack(**new_kp)

        self.descriptors_vectors = self.write_descriptors(self.key_points, self.octave_pyramid)

        return self.key_points, self.descriptors_vectors

    def keys_to_image_size(self, key_points: tuple, unpack_octave: bool = False):
        key_points = self.__key_points_pack(*key_points)
        pt_unpack = key_points.pt * tf.constant([1.0, 0.5, 0.5, 1.0], dtype=tf.float32)
        size_unpack = key_points.size * 0.5
        octave_unpack = bitwise_ops.bitwise_xor(tf.cast(key_points.octave, dtype=tf.int64), 255)

        unpack_key_points = self.__key_points_pack(
            pt_unpack, size_unpack, key_points.angle, tf.cast(octave_unpack, dtype=tf.float32), key_points.octave_id,
            key_points.response
        )
        if not unpack_octave:
            return unpack_key_points

        octave = bitwise_ops.bitwise_and(octave_unpack, 255)
        layer = bitwise_ops.right_shift(octave_unpack, 8)
        layer = bitwise_ops.bitwise_and(layer, 255)
        octave = tf.where(octave >= 128, bitwise_ops.bitwise_or(octave, -128), octave)
        # scale = 1.0 / tf.cast(tf.bitwise.left_shift(1, math_ops.abs(octave)), dtype=tf.float32)
        # scale = tf.where(octave == -1, 1 / scale, scale)
        scale = tf.where(
            octave >= 1,
            tf.cast(1 / tf.bitwise.left_shift(1, octave), dtype=tf.float32),
            tf.cast(tf.bitwise.left_shift(1, -octave), dtype=tf.float32)
        )

        unpacked_octave = namedtuple('unpacked_octave', 'octave, layer, scale')
        unpacked_octave = unpacked_octave(tf.cast(octave, dtype=tf.float32), tf.cast(layer, dtype=tf.float32), scale)
        return unpack_key_points, unpacked_octave

    def compute_max_con(self, points, shapes):
        shapes = tf.cast(shapes, dtype=tf.int32)
        points = tf.cast(points, dtype=tf.int32)

        _, points_h, points_w, _ = tf.unstack(points, 4, axis=-1)
        _, limit_h, limit_w, _ = tf.unstack(shapes, 4, axis=-1)

        valid_h = math_ops.minimum(points_h, limit_h - 1 - points_h)
        valid_w = math_ops.minimum(points_w, limit_w - 1 - points_w)

        max_con = math_ops.minimum(valid_h, valid_w)
        max_con = math_ops.maximum(max_con, 0)
        return max_con

    def compute_parallel_jobs(self, con_shape, octave_ids):
        uni_con = tf.unique(con_shape)
        uni_oct = tf.unique(octave_ids)
        parallel_jobs = []
        parallel_tup = namedtuple('parallel_jobs', 'idx, con, octave_id')

        for ci, c in enumerate(uni_con.y):
            mask = math_ops.equal(uni_con.idx, ci)
            for oi, o in enumerate(uni_oct.y):
                mask_ = math_ops.logical_and(mask, math_ops.equal(uni_oct.idx, oi))
                mask_ = tf.where(mask_)
                if mask_.shape[0] > 0:
                    parallel_jobs.append(parallel_tup(tf.reshape(mask_, (-1,)), int(c), int(o)))
        return parallel_jobs

    def trilinear_interpolation(self, yxz: tf.Tensor, values: tf.Tensor) -> tuple:
        yxz = tf.cast(yxz, dtype=values.dtype)

        y, x, z = tf.unstack(yxz, num=3, axis=-1)

        # interpolation in x direction
        _C0 = values * (1 - x)
        _C1 = values * x

        # interpolation in y direction
        _C00 = _C0 * (1 - y)
        _C01 = _C0 * y

        _C10 = _C1 * (1 - y)
        _C11 = _C1 * y

        # interpolation in z direction
        _C000 = _C00 * (1 - z)
        _C001 = _C00 * z
        _C010 = _C01 * (1 - z)
        _C011 = _C01 * z
        _C100 = _C10 * (1 - z)
        _C101 = _C10 * z
        _C110 = _C11 * (1 - z)
        _C111 = _C11 * z

        points = namedtuple('points', 'C000, C001, C010, C011, C100, C101, C110, C111')
        out = points(_C000, _C001, _C010, _C011, _C100, _C101, _C110, _C111)
        return out

    def unpack_tri_to_histogram(self, yxz: tf.Tensor, points: tuple, window_width: int, n_bins: int = 8) -> tf.Tensor:
        y, x, z = tf.unstack(yxz, num=3, axis=-1)
        histogram = tf.zeros((yxz.shape[0], window_width, window_width, n_bins), dtype=points[0].dtype)

        b = tf.ones([*yxz.shape[:-1]], dtype=tf.int32) * tf.reshape(tf.range(yxz.shape[0], dtype=tf.int32), (-1, 1))
        b = tf.reshape(b, (-1, 1))

        _add_one = lambda c: c + 1
        _add_two = lambda c: c + 2
        _modulus = lambda c: (c + 1) % n_bins
        _none = lambda c: c

        z_pack = [_none, _modulus]
        x_pack = [_add_one, _add_two]
        y_pack = [_add_one, _add_two]

        for i, _C in enumerate(points):
            zi = z_pack[0](z)
            z_pack[0], z_pack[1] = z_pack[1], z_pack[0]

            if (i % 2) == 0: x_pack[0], x_pack[1] = x_pack[1], x_pack[0]
            xi = x_pack[0](x)

            if (i % 4) == 0: y_pack[0], y_pack[1] = y_pack[1], y_pack[0]
            yi = y_pack[0](y)

            mask_ = math_ops.logical_and(
                tf.logical_and(xi >= 0, xi < window_width), tf.logical_and(yi >= 0, yi < window_width)
            )
            mask_ = tf.reshape(mask_, (-1,))

            cords = tf.stack((yi, xi, zi), axis=-1)
            cords = tf.cast(cords, dtype=tf.int32)
            cords = tf.reshape(cords, (-1, 3))
            cords = tf.concat((b, cords), axis=-1)

            cords = tf.boolean_mask(cords, mask_)
            _C = tf.boolean_mask(tf.reshape(_C, (-1,)), mask_)

            histogram = tf.tensor_scatter_nd_add(histogram, cords, _C)
        return histogram

    def write_descriptors(self, key_points: tuple, octaves: list) -> tf.Tensor:
        args = self.descriptors_args
        key_points = self.__key_points_pack(*key_points)
        octaves = [self.__octaves_pack(*oc) for oc in octaves]

        unpack_kp, unpack_oct = self.keys_to_image_size(key_points, unpack_octave=True)

        bins_per_degree = args.n_bins / 360.
        weight_multiplier = -0.5 / ((0.5 * args.window_width) ** 2)

        unpack_oct = unpack_oct._replace(octave=tf.cast(unpack_oct.octave + 1, dtype=tf.int32))
        unpack_oct = unpack_oct._replace(
            octave=tf.where(unpack_oct.octave < 0, len(octaves) + unpack_oct.octave, unpack_oct.octave)
        )
        unpack_oct = unpack_oct._replace(layer=tf.cast(unpack_oct.layer, dtype=tf.int32))

        scale_pad = tf.pad(tf.repeat(unpack_oct.scale, 2, axis=1), paddings=tf.constant([[0, 0], [1, 0]]),
                           constant_values=1.0)
        scale_pad = tf.pad(scale_pad, paddings=tf.constant([[0, 0], [0, 1]]), constant_values=0.0)

        point = tf.cast(tf.round(unpack_kp.pt * scale_pad), dtype=tf.int64)
        point, _ = tf.split(point, [3, 1], axis=-1)
        point = tf.concat((point, tf.cast(unpack_oct.layer, dtype=tf.int64)), axis=-1) - tf.constant([[0, 1, 1, 0]],
                                                                                                     dtype=tf.int64)

        angle = 360. - unpack_kp.angle
        cos_sin_angle = namedtuple('cos_sin_angle', 'cos, sin')
        cos_sin_angle = cos_sin_angle(math_ops.cos((PI / 180) * angle), math_ops.sin((PI / 180) * angle))

        hist_width = args.scale_multiplier * 0.5 * unpack_oct.scale * unpack_kp.size
        hist_con = tf.round(hist_width * math_ops.sqrt(2.0) * (args.window_width + 1.0) * 0.5)
        hist_con = tf.cast(hist_con, dtype=tf.int32)

        hist_con = tf.reshape((hist_con * 2) + 1, (-1,))

        oct_shapes = [tf.reshape(tf.shape(a.dx), (1, 4)) for a in octaves]
        oct_shapes = tf.gather(tf.concat(oct_shapes, axis=0), tf.squeeze(unpack_oct.octave))

        maximum_hist_con = self.compute_max_con(point, oct_shapes)

        con = math_ops.minimum(tf.squeeze(hist_con), maximum_hist_con)
        parallel = self.compute_parallel_jobs(con, tf.squeeze(unpack_oct.octave))

        descriptors = tf.zeros((angle.shape[0], (args.window_width ** 2) * args.n_bins))

        for i_job in parallel:
            if i_job.con == 0:
                continue
            i_octave = octaves[i_job.octave_id]
            i_hist_width = tf.gather(hist_width, i_job.idx)

            i_block = make_neighborhood2D(tf.constant([[0, 0, 0, 0]] * len(i_job.idx), dtype=tf.int64), con=i_job.con)
            i_point = tf.gather(point, i_job.idx)

            i_cords = i_block + tf.expand_dims(i_point, axis=1)

            i_magnitude = tf.gather_nd(i_octave.magnitude, tf.reshape(i_cords, (-1, 4)))
            i_orientation = tf.gather_nd(i_octave.orientation % 360., tf.reshape(i_cords, (-1, 4)))
            i_angle = tf.gather(angle, i_job.idx)

            i_cos = tf.gather(cos_sin_angle.cos, i_job.idx)
            i_sin = tf.gather(cos_sin_angle.sin, i_job.idx)

            i_block = tf.cast(i_block, dtype=tf.float32)
            _, i_y, i_x, _ = tf.unstack(i_block, 4, axis=-1)

            i_orientation = tf.reshape(i_orientation, shape=i_y.shape)
            i_magnitude = tf.reshape(i_magnitude, shape=i_y.shape)

            i_z = (i_orientation - i_angle) * bins_per_degree

            i_yr = i_x * i_sin + i_y * i_cos
            i_xr = i_x * i_cos - i_y * i_sin

            i_y = (i_yr / i_hist_width) + 0.5 * 4 - 0.5
            i_x = (i_xr / i_hist_width) + 0.5 * 4 - 0.5

            i_weight = math_ops.exp(weight_multiplier * ((i_yr / i_hist_width) ** 2 + (i_x / i_hist_width) ** 2))
            i_values = i_magnitude * i_weight

            i_yxz = [i_y, i_x, i_z]

            i_yxz_floor = [math_ops.floor(c) for c in i_yxz]
            i_yxz_frac = [c1 - c2 for c1, c2 in zip(i_yxz, i_yxz_floor)]

            i_yxz_floor[-1] = tf.where(
                math_ops.logical_and(i_yxz_floor[-1] < 0, i_yxz_floor[-1] + args.n_bins < args.n_bins),
                i_yxz_floor[-1] + args.n_bins,
                i_yxz_floor[-1]
            )

            i_yxz_frac = tf.stack(i_yxz_frac, axis=-1)
            i_yxz_floor = tf.stack(i_yxz_floor, axis=-1)

            i_intr = self.trilinear_interpolation(i_yxz_frac, i_values)

            i_histogram = self.unpack_tri_to_histogram(
                points=i_intr, yxz=i_yxz_floor, window_width=args.window_width + 2, n_bins=args.n_bins
            )

            i_descriptor_vector = tf.slice(
                i_histogram, [0, 1, 1, 0],
                [i_histogram.shape[0], args.window_width, args.window_width, args.n_bins]
            )
            i_descriptor_vector = tf.reshape(i_descriptor_vector, (i_histogram.shape[0], -1))

            threshold = tf.norm(i_descriptor_vector) * args.descriptor_max_value

            i_descriptor_vector = tf.where(i_descriptor_vector > threshold, threshold, i_descriptor_vector)
            i_descriptor_vector = i_descriptor_vector / tf.maximum(tf.norm(i_descriptor_vector), self.epsilon)
            i_descriptor_vector = tf.round(i_descriptor_vector * 512)
            i_descriptor_vector = tf.maximum(i_descriptor_vector, 0)
            i_descriptor_vector = tf.minimum(i_descriptor_vector, 255)

            axs = tf.reshape(tf.range(i_descriptor_vector.shape[1], dtype=tf.int64), shape=(-1, 1))
            axs = tf.repeat(axs, i_descriptor_vector.shape[0], axis=0)
            i_id = tf.ones_like(i_descriptor_vector, dtype=tf.int64) * tf.reshape(i_job.idx, (-1, 1))
            i_id = tf.reshape(i_id, (-1, 1))
            axs = tf.concat((i_id, axs), axis=-1)

            descriptors = tf.tensor_scatter_nd_add(descriptors, axs, tf.reshape(i_descriptor_vector, (-1,)))
        return descriptors


def show_key_points(key_points, image):
    from viz import show_images

    cords = tf.cast(key_points.pt, dtype=tf.int64) * tf.constant([1, 1, 1, 0], dtype=tf.int64)

    marks = make_neighborhood2D(cords, con=3, origin_shape=image.shape)
    marks = tf.reshape(marks, shape=(-1, 4))
    marked = tf.scatter_nd(marks, tf.ones(shape=(marks.get_shape()[0],), dtype=tf.float32) * 255.0, shape=image.shape)
    marked_del = math_ops.abs(marked - 255.0) / 255.0
    img_mark = image * marked_del

    marked = tf.concat((marked, tf.zeros_like(image), tf.zeros_like(image)), axis=-1)

    img_mark = img_mark + marked
    show_images(img_mark, 1, 1)


if __name__ == '__main__':
    img1 = tf.keras.utils.load_img('box.png', color_mode='grayscale')
    img1 = tf.convert_to_tensor(tf.keras.utils.img_to_array(img1), dtype=tf.float32)
    img1 = img1[tf.newaxis, ...]
    img1 = tf.repeat(img1, repeats=2, axis=0)

    alg = SIFT(sigma=1.4, n_octaves=4, n_intervals=4, with_descriptors=True)
    kp, disc = alg.call(img1)
