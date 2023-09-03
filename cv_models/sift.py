from typing import Union
import tensorflow as tf
from cv_models.utils import PI, gaussian_kernel, compute_extrema3D, \
    make_neighborhood2D, compute_central_gradient3D, compute_hessian_3D
from tensorflow.python.keras import backend
from collections import namedtuple
from dataclasses import dataclass, field

# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

backend.set_floatx('float32')
linalg_ops = tf.linalg
math_ops = tf.math
bitwise_ops = tf.bitwise
image_ops = tf.image


@dataclass(eq=False, order=False)
class KeyPoints:
    __slots__ = ('pt', 'size', 'angle', 'octave', 'octave_id', 'response')
    pt: tf.Tensor
    size: tf.Tensor
    angle: tf.Tensor
    octave: tf.Tensor
    octave_id: tf.Tensor
    response: tf.Tensor

    def __post_init__(self):
        if not isinstance(self.pt, tf.Tensor):
            raise ValueError('All the fields need to be type of tf.Tensor')
        _shape = self.pt.get_shape().as_list()
        if len(_shape) > 2:
            self.pt = tf.squeeze(self.pt)
            _shape = self.pt.get_shape().as_list()
        if len(_shape) != 2 or _shape[-1] != 4: raise ValueError('expected "pt" to be 2D tensor with size of (None, 4)')
        for f in [self.size, self.angle, self.octave, self.octave_id, self.response]:
            if not isinstance(f, tf.Tensor) or f.shape[0] != _shape[0]:
                raise ValueError('All the fields need to be type of Tensor with the same first dim size')

    @property
    def shape(self):
        _len = (self.pt.shape[0],)
        return _len

    def to_image_size(self, unpack_octave: bool = False):
        pt_unpack = self.pt * tf.constant([1.0, 0.5, 0.5, 1.0], dtype=tf.float32)
        size_unpack = self.size * 0.5
        octave_unpack = bitwise_ops.bitwise_xor(tf.cast(self.octave, dtype=tf.int64), 255)

        unpack_key_points = KeyPoints(
            pt_unpack, size_unpack, self.angle, tf.cast(octave_unpack, dtype=tf.float32), self.octave_id, self.response
        )
        if not unpack_octave:
            return unpack_key_points

        octave = bitwise_ops.bitwise_and(octave_unpack, 255)
        layer = bitwise_ops.right_shift(octave_unpack, 8)
        layer = bitwise_ops.bitwise_and(layer, 255)
        octave = tf.where(octave >= 128, bitwise_ops.bitwise_or(octave, -128), octave)
        scale = tf.where(
            octave >= 0,
            tf.cast(1 / tf.bitwise.left_shift(1, octave), dtype=tf.float32),
            tf.cast(tf.bitwise.left_shift(1, -octave), dtype=tf.float32)
        )
        unpacked_octave = namedtuple('unpacked_octave', 'octave, layer, scale')
        unpacked_octave = unpacked_octave(tf.cast(octave, dtype=tf.float32), tf.cast(layer, dtype=tf.float32), scale)
        return unpack_key_points, unpacked_octave

    def stack(self):
        _stack = tf.concat(([self.pt, self.size, self.angle, self.octave, self.octave_id, self.response]), axis=-1)
        return _stack


@dataclass(eq=False, order=False)
class Octave:
    __slots__ = ('gaussian', 'dx', 'dy', 'magnitude', 'orientation')
    gaussian: tf.Tensor
    dx: tf.Tensor
    dy: tf.Tensor
    magnitude: tf.Tensor
    orientation: tf.Tensor

    def __post_init__(self):
        if not isinstance(self.gaussian, tf.Tensor):
            raise ValueError('All the fields need to be type of Tensor')
        _batch_shape = self.gaussian.get_shape()[0]
        for X in [self.dx, self.dy, self.magnitude, self.orientation]:
            if not isinstance(X, tf.Tensor) or X.get_shape()[0] != _batch_shape:
                raise ValueError('All the fields need to be type of Tensor with same batch size')

    @property
    def shape(self):
        return self.gaussian.shape.as_list()


@dataclass(eq=False, order=False)
class Argumentor:
    sigma: float = 1.4
    assume_blur_sigma: float = 0.5
    n_intervals: int = 4
    n_octaves: int = 4
    border_width: tuple = (5, 5, 0)
    convergence_N: int = field(default=5, init=False)
    orientation_N_bins: int = field(default=36, init=False)
    eigen_ration: int = field(default=10, init=False)
    peak_ratio: float = field(default=0.8, init=False)
    contrast_threshold: float = field(default=0.04, init=False)
    scale_factor: float = field(default=1.5, init=False)
    extrema_offset: float = field(default=0.5, init=False)
    radius_factor: int = field(default=3, init=False)
    con: int = field(default=3, init=False)
    descriptors_N_bins: int = field(default=8, init=False)
    window_width: int = field(default=4, init=False)
    scale_multiplier: int = field(default=3, init=False)
    descriptor_max_value: float = field(default=0.2, init=False)


class SIFT:

    def __init__(self,
                 sigma: float = 1.4, assume_blur_sigma: float = 0.5, n_intervals: int = 4,
                 n_octaves: Union[int, None] = None, name: Union[str, None] = None):
        self.__inputs_shape = None
        self.name = name or 'SIFT'
        self.epsilon = 1e-07
        self.octave_pyramid = []
        self.key_points = None
        self.descriptors_vectors = None
        self.graph_args = Argumentor(
            sigma=sigma, assume_blur_sigma=assume_blur_sigma, n_intervals=n_intervals, n_octaves=n_octaves
        )
        self.kernels = namedtuple('kernels', 'base, gaussian, gradient')

    def __call__(self, inputs):
        return self.call(inputs)

    def __validate_input(self, inputs: tf.Tensor) -> tf.Tensor:
        _shape = inputs.get_shape().as_list()
        _ndims = len(_shape)
        if _ndims != 4 or _shape[-1] != 1:
            raise ValueError(
                'expected the inputs to be grayscale images with size of (None, h, w, 1)'
            )
        self.__inputs_shape = _shape
        inputs = tf.cast(inputs, dtype=tf.float32)
        return inputs

    def __init_graph(self):
        _, h_, w_, _ = self.__inputs_shape
        args = self.graph_args
        self.octave_pyramid = []
        self.key_points = None
        delta_sigma = (args.sigma ** 2) - ((2 * args.assume_blur_sigma) ** 2)
        delta_sigma = math_ops.sqrt(tf.maximum(delta_sigma, 0.64))

        base_kernel = gaussian_kernel(kernel_size=0, sigma=delta_sigma)
        base_kernel = tf.expand_dims(tf.expand_dims(base_kernel, axis=-1), axis=-1)

        images_per_octaves = args.n_intervals + 3
        K = 2 ** (1 / args.n_intervals)
        K = tf.cast(K, dtype=tf.float32)

        sigma_prev = (K ** tf.cast(tf.range(images_per_octaves - 1), dtype=tf.float32)) * args.sigma
        sigmas = math_ops.sqrt((K * sigma_prev) ** 2 - sigma_prev ** 2)
        sigmas = math_ops.maximum(sigmas, 0.8)
        gaussian_kernels = []

        for i, s in enumerate(sigmas):
            kernel_ = gaussian_kernel(kernel_size=0, sigma=s)
            gaussian_kernels.append(tf.expand_dims(tf.expand_dims(kernel_, axis=-1), axis=-1))

        kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], shape=(3, 3, 1, 1), dtype=tf.float32)
        ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], shape=(3, 3, 1, 1), dtype=tf.float32)
        gradient_kernel = tf.concat((kx, ky), axis=-1)

        min_shape = int(gaussian_kernels[-1].get_shape()[0])
        max_n_octaves = self.compute_default_N_octaves(height=h_ * 2, weight=w_ * 2, min_shape=min_shape)

        if args.n_octaves is not None and max_n_octaves > args.n_octaves:
            max_n_octaves = args.n_octaves
        args.n_octaves = max_n_octaves

        self.kernels = namedtuple('kernels', 'base, gaussian, gradient')(base_kernel, gaussian_kernels, gradient_kernel)

    def __localize_extrema(self, octave: Octave) -> tuple:
        args = self.graph_args
        dim = octave.shape[-1]

        dog = math_ops.subtract(tf.split(octave.gaussian, [1, dim - 1], -1)[1],
                                tf.split(octave.gaussian, [dim - 1, 1], -1)[0])
        dog_shape = dog.get_shape().as_list()

        extrema = compute_extrema3D(dog, con=args.con, border_width=args.border_width)

        dog = self.__scale(tf.abs(dog), -1, 1)
        grad = compute_central_gradient3D(dog)
        grad = tf.expand_dims(grad, -1)

        hess = compute_hessian_3D(dog)
        # TODO: the problem is here!!!
        extrema_update = - linalg_ops.lstsq(hess, grad, l2_regularizer=0.0, fast=False)
        extrema_update = tf.squeeze(extrema_update, axis=-1)

        dot_ = tf.reduce_sum(
            tf.multiply(tf.expand_dims(extrema_update, 3), tf.transpose(grad, perm=(0, 1, 2, 5, 3, 4))),
            axis=-1, keepdims=False
        )
        dot_ = tf.squeeze(dot_, 3)

        mid_cube_values = tf.slice(dog, [0, 1, 1, 1],
                                   [dog_shape[0], dog_shape[1] - 2, dog_shape[2] - 2, dog_shape[3] - 2])
        update_response = mid_cube_values + 0.5 * dot_

        hess_shape = hess.get_shape().as_list()
        hess_xy = tf.slice(hess, [0, 0, 0, 0, 0, 0], [*hess_shape[:-2], 2, 2])
        hess_xy_trace = linalg_ops.trace(hess_xy)
        hess_xy_det = linalg_ops.det(hess_xy)

        kp_cond1 = math_ops.less(math_ops.reduce_max(math_ops.abs(extrema_update), axis=-1), args.extrema_offset)

        kp_cond2 = math_ops.greater_equal(
            math_ops.abs(update_response) * int(args.n_intervals), args.contrast_threshold
        )

        kp_cond3 = math_ops.logical_and(
            math_ops.greater(hess_xy_det, 0.0),
            math_ops.less(args.eigen_ration * (hess_xy_trace ** 2), ((args.eigen_ration + 1) ** 2) * hess_xy_det)
        )
        cond = math_ops.logical_and(kp_cond1, math_ops.logical_and(kp_cond2, kp_cond3))

        kp_cond4 = tf.scatter_nd(extrema, tf.ones((extrema.shape[0],), dtype=tf.bool), dog_shape)
        kp_cond4 = tf.slice(kp_cond4, [0, 1, 1, 1],
                            [dog_shape[0], dog_shape[1] - 2, dog_shape[2] - 2, dog_shape[3] - 2])
        tup = namedtuple('extrema', 'cond, extrema, extrema_update, response')
        tup = tup(cond, kp_cond4, extrema_update, update_response)
        return tup

    def __compute_histogram(self, key_points: KeyPoints, octaves: list[Octave]) -> tf.Tensor:
        args = self.graph_args

        histogram = tf.zeros(shape=(key_points.pt.shape[0], args.orientation_N_bins), dtype=tf.float32)

        scale = args.scale_factor * key_points.size / (2 ** (key_points.octave_id + 1))

        radius = tf.cast(tf.round(args.radius_factor * scale), dtype=tf.int32)
        weight_factor = -0.5 / (scale ** 2)

        _prob = 1.0 / (2 ** key_points.octave_id)
        _one = tf.ones_like(_prob)
        _prob = tf.stack((_one, _prob, _prob, _one), axis=-1)
        _prob = tf.squeeze(_prob)

        region_center = tf.cast(key_points.pt * _prob, dtype=tf.int64) - tf.constant([[0, 1, 1, 0]], dtype=tf.int64)

        con = tf.reshape((radius * 2) + 1, (-1,))
        octave_id = tf.reshape(tf.cast(key_points.octave_id, tf.int32), (-1,))

        oct_shapes = [tf.reshape(a.shape, shape=(1, 4)) for a in octaves]
        oct_shapes = tf.gather(tf.concat(oct_shapes, axis=0), octave_id)

        maximum_con = self.__compute_max_con(region_center, oct_shapes)
        con = math_ops.minimum(con, maximum_con)
        parallel = self.__compute_parallel_jobs(con, octave_id)

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

            curr_hist_index = tf.cast(tf.round(curr_orientation * args.orientation_N_bins / 360.), dtype=tf.int64)
            curr_hist_index = curr_hist_index % args.orientation_N_bins

            curr_hist_id = tf.ones_like(curr_weight, dtype=tf.int64) * tf.reshape(curr_parallel.idx, (-1, 1))
            curr_weight = tf.reshape(curr_weight, (-1,))

            curr_hist_index = tf.concat((tf.reshape(curr_hist_id, (-1, 1)), tf.reshape(curr_hist_index, (-1, 1))),
                                        axis=-1)

            histogram = tf.tensor_scatter_nd_add(histogram, curr_hist_index, curr_weight * curr_magnitude)

        gaussian1D = tf.constant([1, 4, 6, 4, 1], dtype=tf.float32) / 16.0
        gaussian1D = tf.reshape(gaussian1D, shape=(-1, 1, 1))

        histogram_pad = tf.pad(
            tf.expand_dims(histogram, axis=-1), paddings=tf.constant([[0, 0], [2, 2], [0, 0]], dtype=tf.int32),
            constant_values=0.0
        )

        smooth_histogram = tf.nn.convolution(histogram_pad, gaussian1D, padding='VALID')
        smooth_histogram = tf.squeeze(smooth_histogram, axis=-1)

        orientation_max = tf.reduce_max(smooth_histogram, axis=-1)
        value_cond = tf.repeat(tf.reshape(orientation_max, shape=(-1, 1)), repeats=args.orientation_N_bins, axis=-1)
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

        left_index = (p_idx - 1) % args.orientation_N_bins
        right_index = (p_idx + 1) % args.orientation_N_bins

        left_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, left_index), axis=-1))
        right_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, right_index), axis=-1))

        interpolated_peak_index = ((tf.cast(p_idx, dtype=tf.float32) + 0.5 * (left_value - right_value)) / (
                left_value - (2 * peak_value) + right_value)) % args.orientation_N_bins

        orientation = 360. - interpolated_peak_index * 360. / args.orientation_N_bins

        orientation = tf.where(math_ops.less(math_ops.abs(orientation), 1e-7), 0.0, orientation)

        p_id = tf.reshape(p_id, (-1,))

        prev_key_stack = key_points.stack()
        prev_key_stack = tf.gather(prev_key_stack, p_id)
        prev_key_split = tf.split(prev_key_stack, [4, 1, 1, 1, 1, 1], axis=-1)

        key_points = KeyPoints(
            pt=prev_key_split[0],
            size=prev_key_split[1],
            angle=tf.reshape(orientation, (-1, 1)),
            octave=prev_key_split[3],
            octave_id=prev_key_split[4],
            response=prev_key_split[5]
        )

        return histogram, key_points

    def __compute_max_con(self, points, shapes):
        shapes = tf.cast(shapes, dtype=tf.int32)
        points = tf.cast(points, dtype=tf.int32)

        _, points_h, points_w, _ = tf.unstack(points, 4, axis=-1)
        _, limit_h, limit_w, _ = tf.unstack(shapes, 4, axis=-1)

        valid_h = math_ops.minimum(points_h, limit_h - 2 - points_h)
        valid_w = math_ops.minimum(points_w, limit_w - 2 - points_w)

        max_con = math_ops.minimum(valid_h, valid_w)
        max_con = math_ops.maximum(max_con, 0)
        return max_con

    def __compute_parallel_jobs(self, con_shape, octave_ids):
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

    def __remove_duplicates(self, key_points: KeyPoints) -> KeyPoints:
        def map_1(args):
            i = 0
            for itm in args:
                if itm != 0: break
                i += 1
            return args[i] if i != len(args) else args[i - 1]

        def map_2(args):
            arg1, arg2 = tf.split(args, [1, 1], -1)
            if tf.reduce_max(arg2 - arg1) == 0:
                return False
            return True

        diff_kernel = tf.constant([-1, 1], shape=(2, 1, 1, 1), dtype=tf.float32)
        kp_stack = [key_points.pt, key_points.size, key_points.angle, key_points.response, key_points.octave,
                    key_points.octave_id]
        kp_stack = tf.concat(kp_stack, axis=-1)

        partitions = tf.split(key_points.pt, [1, 3], axis=-1)[0]
        partitions = tf.cast(tf.squeeze(partitions), tf.int32)

        split_batch = tf.dynamic_partition(kp_stack, partitions, tf.reduce_max(partitions) + 1)

        clean_kp = None

        for curr_batch in split_batch:
            curr_diff = tf.nn.convolution(tf.expand_dims(tf.expand_dims(curr_batch, -1), 0), diff_kernel,
                                          padding='VALID')
            curr_diff = tf.squeeze(curr_diff)

            curr_diff = tf.map_fn(map_1, curr_diff)

            values, indices = math_ops.top_k(curr_diff, k=len(curr_diff))

            curr_compare = tf.split(curr_batch, [6, 3], axis=-1)[0]
            first_ = tf.expand_dims(tf.gather(curr_compare, indices[:-1]), -1)
            next_ = tf.expand_dims(tf.gather(curr_compare, indices[1:]), -1)

            curr_map = tf.map_fn(map_2, tf.concat((first_, next_), axis=-1))

            index_in = tf.boolean_mask(indices[1:], curr_map)
            index_in = tf.concat((tf.reshape(indices[0], (1,)), index_in), axis=0)
            curr_clean = tf.gather(curr_batch, index_in)
            if clean_kp is None:
                clean_kp = curr_clean
            else:
                clean_kp = tf.concat((clean_kp, curr_clean), axis=0)
        pt, size, angle, response, octave, octave_id = tf.split(clean_kp, [4, 1, 1, 1, 1, 1], axis=-1)
        out = KeyPoints(pt=pt, size=size, angle=angle, response=response, octave=octave, octave_id=octave_id)
        return out

    def __trilinear_interpolation(self, yxz: tf.Tensor, values: tf.Tensor) -> tuple:
        yxz = tf.cast(yxz, dtype=values.dtype)

        y, x, z = tf.unstack(yxz, num=3, axis=-1)

        # interpolation in y direction
        _C0 = values * (1 - y)
        _C1 = values * y

        # interpolation in x direction
        _C00 = _C0 * (1 - x)
        _C01 = _C0 * x

        _C10 = _C1 * (1 - x)
        _C11 = _C1 * x

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

    def __unpack_tri_to_histogram(self, yxz: tf.Tensor, points: tuple, window_width: int, n_bins: int = 8) -> tf.Tensor:
        y, x, z = tf.unstack(yxz, num=3, axis=-1)
        histogram = tf.zeros((yxz.shape[0], window_width, window_width, n_bins), dtype=points[0].dtype)

        b = tf.ones([*yxz.shape[:-1]], dtype=tf.int32) * tf.reshape(tf.range(yxz.shape[0], dtype=tf.int32), (-1, 1))
        b = tf.reshape(b, (-1, 1))

        _add_one = lambda c: c + 1
        _add_two = lambda c: c + 2
        _modulus = lambda c: (c + 1) % n_bins
        _none = lambda c: c

        z_pack = [_none, _modulus]
        x_pack = [_add_two, _add_one]
        y_pack = [_add_two, _add_one]

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

    def __scale(self, X: tf.Tensor, min_val: int = 0, max_val: int = 1) -> tf.Tensor:
        return ((X - tf.reduce_min(X)) / (tf.reduce_max(X) - tf.reduce_min(X))) * (max_val - min_val) + min_val

    def call(self, inputs: tf.Tensor) -> tuple[KeyPoints, tf.Tensor]:
        inputs = self.__validate_input(inputs)
        self.__init_graph()
        _, h_, w_, _ = self.__inputs_shape
        args = self.graph_args

        inputs = self.__scale(inputs, 0, 255)
        with tf.name_scope('Xbase'):
            X_base = image_ops.resize(inputs, size=[h_ * 2, w_ * 2], method='bilinear', name='Xb_up')
            X_base = tf.nn.conv2d(X_base, self.kernels.base, strides=[1, 1, 1, 1], padding='SAME', name='Xb_blur')

        with tf.name_scope('octave_pyramid'):
            gaussian_kernels = self.kernels.gaussian
            grad_kernels = self.kernels.gradient

            X = X_base
            size_ = [h_, w_]
            key_points_warp = tf.constant([[]], shape=(0, 9), dtype=tf.float32)
            for oc_id in range(args.n_octaves):
                gauss_cap = [X]
                grad = tf.expand_dims(tf.nn.convolution(X, grad_kernels, padding='VALID'), axis=3)
                grad_cap = [grad]

                for kernel in gaussian_kernels:
                    X = tf.nn.convolution(X, kernel, padding='SAME')
                    grad = tf.expand_dims(tf.nn.convolution(X, grad_kernels, padding='VALID'), axis=3)
                    gauss_cap.append(X)
                    grad_cap.append(grad)

                if oc_id < args.n_octaves - 1:
                    X = image_ops.resize(gauss_cap[-3], size=size_, method='nearest')
                    size_[0] //= 2
                    size_[1] //= 2

                gauss_cap = tf.concat(gauss_cap, axis=-1)
                grad_cap = tf.concat(grad_cap, axis=3)

                dx, dy = tf.unstack(grad_cap, 2, axis=-1)
                magnitude = math_ops.sqrt(dx * dx + dy * dy)
                orientation = math_ops.atan2(dx, dy) * (180.0 / PI)
                oc = Octave(gauss_cap, dx, dy, magnitude, orientation)
                self.octave_pyramid.append(oc)

                with tf.name_scope('localize_extrema'):
                    extrema_oc = self.__localize_extrema(oc)
                    sure_key_points = math_ops.logical_and(extrema_oc.cond, extrema_oc.extrema)
                    shape_ = extrema_oc.extrema.shape
                    attempts = math_ops.logical_and(extrema_oc.cond, ~sure_key_points)

                    for _ in range(args.convergence_N):
                        attempts_cords = tf.where(attempts)
                        if attempts_cords.shape[0] == 0: break
                        attempts_cords = tf.reshape(attempts_cords, (-1, 4))
                        attempts_update = tf.gather_nd(extrema_oc.extrema_update, attempts_cords)

                        ex, ey, ez = tf.unstack(attempts_update, num=3, axis=-1)
                        cd, cy, cx, cz = tf.unstack(tf.cast(attempts_cords, tf.float32), num=4, axis=1)
                        attempts_next = [cd, cy + ey, cx + ex, cz + ez]

                        cond_next = tf.where(
                            (attempts_next[1] >= 0) & (attempts_next[1] < shape_[1]) & (attempts_next[2] > 0) & (
                                    attempts_next[2] < shape_[2]) & (attempts_next[3] > 0) & (
                                    attempts_next[3] < shape_[3]))

                        attempts_next = tf.stack(attempts_next, -1)
                        attempts_next = tf.cast(tf.gather(attempts_next, tf.squeeze(cond_next)), dtype=tf.int64)
                        if attempts_next.shape[0] == 0: break
                        attempts_next = tf.reshape(attempts_next, (-1, 4))
                        attempts_mask = tf.scatter_nd(attempts_next, tf.ones((attempts_next.shape[0],), dtype=tf.bool),
                                                      shape_)

                        new_cords = tf.where(attempts_mask & ~sure_key_points & extrema_oc.cond)

                        sure_key_points = tf.tensor_scatter_nd_update(sure_key_points, new_cords,
                                                                      tf.ones((new_cords.shape[0],), dtype=tf.bool))

                        attempts = math_ops.logical_and(attempts_mask, ~sure_key_points)

                    cords = tf.where(sure_key_points)
                    if cords.shape[0] == 0: continue
                    kp_cords = cords + tf.constant([[0, 1, 1, 1]], dtype=tf.int64)

                    kp_extrema_update = tf.gather_nd(extrema_oc.extrema_update, cords)

                    octave_index = tf.cast(oc_id, dtype=tf.float32)

                    ex, ey, ez = tf.unstack(kp_extrema_update, num=3, axis=1)
                    cd, cy, cx, cz = tf.unstack(tf.cast(kp_cords, tf.float32), num=4, axis=1)

                    kp_pt = tf.stack(
                        (cd, (cy + ey) * (2 ** octave_index), (cx + ex) * (2 ** octave_index), cz), axis=-1
                    )

                    kp_octave = octave_index + cz * (2 ** 8) + tf.round((ez + 0.5) * 255.0) * (2 ** 16)

                    kp_size = args.sigma * (2 ** ((cz + ez) / tf.cast(args.n_intervals, dtype=tf.float32))) * (
                            2 ** (octave_index + 1.0))
                    kp_response = math_ops.abs(tf.gather_nd(extrema_oc.response, cords))

                    octave_index = tf.ones_like(kp_size) * octave_index
                    angle = tf.ones_like(kp_size) * -1.0
                    key_point_curr = [
                        tf.reshape(kp_pt, (-1, 4)),
                        tf.reshape(kp_size, (-1, 1)),
                        tf.reshape(angle, (-1, 1)),
                        tf.reshape(kp_octave, (-1, 1)),
                        tf.reshape(octave_index, (-1, 1)),
                        tf.reshape(kp_response, (-1, 1))

                    ]
                    key_points_warp = tf.concat((key_points_warp, tf.concat(key_point_curr, -1)), axis=0)

        key_points = KeyPoints(*tf.split(key_points_warp, [4, 1, 1, 1, 1, 1], axis=-1))
        with tf.name_scope('compute_orientations'):
            histogram, key_points = self.__compute_histogram(key_points, self.octave_pyramid)
            self.key_points = self.__remove_duplicates(key_points)
        with tf.name_scope('descriptors_vectors'):
            self.descriptors_vectors = self.write_descriptors(self.key_points, self.octave_pyramid)
        return self.key_points, self.descriptors_vectors

    def write_descriptors(self, key_points: KeyPoints, octaves: list[Octave]) -> tf.Tensor:
        def recompute(oc):
            oc.orientation = oc.orientation % 360
            return oc

        args = self.graph_args
        octaves = [recompute(oc) for oc in octaves]

        unpack_kp, unpack_oct = key_points.to_image_size(unpack_octave=True)

        bins_per_degree = args.descriptors_N_bins / 360.
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

        maximum_hist_con = self.__compute_max_con(point, oct_shapes)

        con = math_ops.minimum(tf.squeeze(hist_con), maximum_hist_con)
        parallel = self.__compute_parallel_jobs(con, tf.squeeze(unpack_oct.octave))

        descriptors = tf.zeros((angle.shape[0], (args.window_width ** 2) * args.descriptors_N_bins))

        for i_job in parallel:
            if i_job.con == 0:
                continue
            i_octave = octaves[i_job.octave_id]
            i_hist_width = tf.gather(hist_width, i_job.idx)

            i_block = make_neighborhood2D(tf.constant([[0, 0, 0, 0]] * len(i_job.idx), dtype=tf.int64), con=i_job.con)
            i_point = tf.gather(point, i_job.idx)

            i_cords = i_block + tf.expand_dims(i_point, axis=1)

            i_magnitude = tf.gather_nd(i_octave.magnitude, tf.reshape(i_cords, (-1, 4)))
            i_orientation = tf.gather_nd(i_octave.orientation, tf.reshape(i_cords, (-1, 4)))
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

            i_y = (i_yr / i_hist_width) + 0.5 * args.window_width - 0.5
            i_x = (i_xr / i_hist_width) + 0.5 * args.window_width - 0.5

            i_weight = math_ops.exp(weight_multiplier * (((i_yr / i_hist_width) ** 2) + (i_x / i_hist_width) ** 2))
            i_values = i_magnitude * i_weight

            i_yxz = [i_y, i_x, i_z]

            i_yxz_floor = [math_ops.floor(c) for c in i_yxz]
            i_yxz_frac = [c1 - c2 for c1, c2 in zip(i_yxz, i_yxz_floor)]

            i_yxz_floor[-1] = tf.where(
                math_ops.logical_and(i_yxz_floor[-1] < 0,
                                     i_yxz_floor[-1] + args.descriptors_N_bins < args.descriptors_N_bins),
                i_yxz_floor[-1] + args.descriptors_N_bins,
                i_yxz_floor[-1]
            )

            i_yxz_frac = tf.stack(i_yxz_frac, axis=-1)
            i_yxz_floor = tf.stack(i_yxz_floor, axis=-1)

            i_intr = self.__trilinear_interpolation(i_yxz_frac, i_values)

            i_histogram = self.__unpack_tri_to_histogram(yxz=i_yxz_floor, points=i_intr,
                                                         window_width=args.window_width + 2,
                                                         n_bins=args.descriptors_N_bins)

            i_descriptor_vector = tf.slice(
                i_histogram, [0, 1, 1, 0],
                [i_histogram.shape[0], args.window_width, args.window_width, args.descriptors_N_bins]
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

    def split_by_batch(self, key_points: KeyPoints, descriptors: Union[tf.Tensor, None] = None) -> Union[
        KeyPoints, tuple[KeyPoints, tf.Tensor]]:
        kp_stack = key_points.stack()
        partitions = tf.split(key_points.pt, [1, 3], axis=-1)[0]
        partitions = tf.cast(tf.squeeze(partitions), tf.int32)

        split_batch = tf.dynamic_partition(kp_stack, partitions, tf.reduce_max(partitions) + 1)
        out = []
        for batch in split_batch:
            curr_kp = KeyPoints(*tf.split(batch, [4, 1, 1, 1, 1, 1], axis=-1))
            out.append(curr_kp)

        if descriptors is None: return out

        descriptors_split = tf.dynamic_partition(descriptors, partitions, tf.reduce_max(partitions) + 1)
        return out, descriptors_split

    @staticmethod
    def compute_default_N_octaves(height: int, weight: int, min_shape: int = 0) -> int:
        s_ = tf.cast(min([height, weight]), dtype=tf.float32)
        diff = math_ops.log(s_)
        if min_shape > 1:
            diff = diff - math_ops.log(tf.cast(min_shape, dtype=tf.float32))

        n_octaves = tf.round(diff / math_ops.log(2.0)) + 1
        return int(n_octaves)


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


def templet_matching(tmp_img, dst_img, tmp_kp, dst_kp, tmp_dsc, dst_dsc):
    import cv2
    import numpy as np
    from viz import show_images

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    tmp_dsc = tmp_dsc.numpy()
    dst_dsc = dst_dsc.numpy()

    matches = flann.knnMatch(tmp_dsc, dst_dsc, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < 10:
        print('number of matches smaller then 10')
        return

    src_index = [m.queryIdx for m in good]
    dst_index = [m.trainIdx for m in good]

    src_pt = tf.gather(tmp_kp.to_image_size().pt, tf.constant(src_index, dtype=tf.int32))
    dst_pt = tf.gather(dst_kp.to_image_size().pt, tf.constant(dst_index, dtype=tf.int32))

    src_pt = tf.split(src_pt, [1, 2, 1], -1)[1].numpy().astype(int).reshape(-1, 1, 2)
    dst_pt = tf.split(dst_pt, [1, 2, 1], -1)[1].numpy().astype(int).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pt, dst_pt, cv2.RANSAC, 5.0)[0]

    _, h, w, _ = tmp_img.shape

    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    dst_img = cv2.polylines(tf.squeeze(dst_img).numpy().astype('uint8'), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    _, h1, w1, _ = tmp_img.shape
    h2, w2 = dst_img.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = tf.squeeze(tmp_img).numpy().astype('uint8')
        newimg[:h2, w1:w1 + w2, i] = dst_img

    for i in range(src_pt.shape[0]):
        pt1 = (int(src_pt[i, 0, 1]), int(src_pt[i, 0, 0] + hdif))
        pt2 = (int(dst_pt[i, 0, 1] + w1), int(dst_pt[i, 0, 0]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))
    show_images([newimg], 1, 1)


if __name__ == '__main__':
    img1 = tf.keras.utils.load_img('box.png', color_mode='grayscale')
    img1 = tf.convert_to_tensor(tf.keras.utils.img_to_array(img1), dtype=tf.float32)
    img1 = img1[tf.newaxis, ...]

    img2 = tf.keras.utils.load_img('box_in_scene.png', color_mode='grayscale')
    img2 = tf.convert_to_tensor(tf.keras.utils.img_to_array(img2), dtype=tf.float32)
    img2 = img2[tf.newaxis, ...]

    alg = SIFT(sigma=1.6, n_intervals=3)
    kp1, disc1 = alg(img1)
    # kp2, disc2 = alg(img2)

    # show_key_points(kp1.to_image_size(), img1)
    # show_key_points(kp2.to_image_size(), img2)

    # templet_matching(img1, img2, kp1, kp2, disc1, disc2)

    # oc = alg.octave_pyramid
    #
    # from viz import show_images
    #
    # g = oc[2].gaussian
    # show_images(tf.transpose(g, (3, 1, 2, 0)), 3, 2)
    #
    # dog = math_ops.subtract(tf.split(g, [1, 6 - 1], -1)[1],
    #                         tf.split(g, [6 - 1, 1], -1)[0])
    # show_images(tf.transpose(((dog - tf.reduce_min(dog)) / (tf.reduce_max(dog) - tf.reduce_min(dog))), (3, 1, 2, 0)), 3, 2)