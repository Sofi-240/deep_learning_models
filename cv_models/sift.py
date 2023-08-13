from typing import Union
import tensorflow as tf
from cv_models.utils import PI, gaussian_kernel, compute_extrema3D, make_neighborhood3D, \
    make_neighborhood2D, gaussian_blur, compute_central_gradient3D, compute_central_gradient2D, compute_hessian_3D
from tensorflow.python.keras import backend
import numpy as np
from collections import namedtuple

# from viz import show_images


math = tf.math
linalg = tf.linalg
bitwise_ops = tf.bitwise

backend.set_floatx('float32')
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf


Octave = namedtuple('Octave', 'octave_id, gaus_X, dog_X')


class KeyPointsSift:
    __DTYPE = backend.floatx()

    def __init__(self, name: Union[str, None] = None, as_size_image: bool = False, initial_size=None):
        self.name = name or 'KeyPointsSift'
        self.pt = None
        self.size = None
        self.angle = None
        self.octave = None
        self.octave_id = None
        self.response = None
        self.__key_points = np.array([], dtype=object)
        self.size_image = as_size_image
        self.initial_size = initial_size

    def __len__(self):
        if self.pt is None:
            return 0
        return int(self.pt.get_shape()[0])

    def __getitem__(self, indices: Union[int, slice, list, tuple]):
        if not isinstance(indices, (int, slice)) or (
                isinstance(indices, (list, tuple)) and not isinstance(indices[0], (list, tuple))):
            raise KeyError(f'{indices}')
        return self.key_points[indices]

    @property
    def __backend(self):
        arrays = [
            self.pt, self.size, self.angle, self.octave, self.octave_id, self.response
        ]
        return arrays

    @__backend.setter
    def __backend(self, values):
        self.pt, self.size, self.angle, self.octave, self.octave_id, self.response = values

    @property
    def key_points(self):
        _len = len(self)
        _keys_len = len(self.__key_points)
        if _len == _keys_len:
            return self.__key_points
        diff = _len - _keys_len
        split_index = [_keys_len] + [1] * diff
        backend_arr = self.__backend
        splits = [
            tf.split(arr, split_index, axis=0)[1:] for arr in backend_arr
        ]
        for index, split in enumerate(zip(*splits)):
            key = self.KeyPoint(
                pt=split[0], size=split[1], angle=split[2], octave=split[3], octave_id=split[4], response=split[5],
                as_size_image=self.size_image
            )
            self.__key_points = np.append(self.__key_points, key)
        return self.__key_points

    def __concat_with_backend(self, values):
        backend_values = self.__backend
        backend_ret = [
            tf.concat((bv, v), axis=0) for bv, v in zip(backend_values, values)
        ]
        self.__backend = backend_ret

    def __pack_backend(self):
        pt, size, angle, octave, octave_id, response = self.__backend

        octave = tf.cast(octave, dtype=self.__DTYPE)
        octave_id = tf.cast(octave_id, dtype=self.__DTYPE)

        concat_ = [pt] + [tf.reshape(a, (-1, 1)) for a in [size, angle, octave, octave_id, response]]

        arr = tf.concat(concat_, axis=1)
        return arr

    def __unpack_backend(self, arr, set_backend=True):
        pt, size, angle, octave, octave_id, response = tf.split(arr, [4, 1, 1, 1, 1, 1], axis=-1)
        octave = tf.cast(octave, dtype=tf.int32)
        octave_id = tf.cast(octave_id, dtype=tf.int32)
        ret = [pt] + [tf.reshape(a, (-1,)) for a in [size, angle, octave, octave_id, response]]
        if set_backend:
            self.__backend = [pt] + [tf.reshape(a, (-1,)) for a in [size, angle, octave, octave_id, response]]
        else:
            return ret

    def add_keys(
            self,
            pt: Union[tf.Tensor, list, tuple],
            size: Union[tf.Tensor, float],
            angle: Union[tf.Tensor, float] = 0.0,
            octave: Union[tf.Tensor, int] = 0,
            octave_id: Union[tf.Tensor, int] = 0,
            response: Union[tf.Tensor, float] = -1.0
    ):

        size = tf.cast(tf.reshape(size, shape=(-1,)), dtype=self.__DTYPE)
        n_points_ = int(tf.shape(size)[0])
        pt = tf.cast(tf.reshape(pt, shape=(-1, 4)), dtype=self.__DTYPE)
        if int(tf.shape(pt)[0]) != n_points_:
            raise ValueError

        def map_args(arg):
            if not isinstance(arg, tf.Tensor):
                return tf.constant([arg] * n_points_)
            arg = tf.reshape(arg, shape=(-1,))
            shape_ = tf.shape(arg)[0]
            if shape_ == 1 and n_points_ > 1:
                return tf.repeat(arg, repeats=n_points_, axis=0)
            assert arg.shape[0] == n_points_
            return arg

        args = [angle, octave, octave_id, response]
        angle, octave, octave_id, response = list(map(map_args, args))

        angle = tf.cast(angle, dtype=self.__DTYPE)
        octave = tf.cast(octave, dtype=tf.int32)
        octave_id = tf.cast(octave_id, dtype=tf.int32)
        response = tf.cast(response, dtype=self.__DTYPE)

        wrap = [pt, size, angle, octave, octave_id, response]
        if self.pt is None:
            self.__backend = wrap
        else:
            self.__concat_with_backend(wrap)

    def remove_duplicate(self):
        _kp = self.key_points
        flatten = [
            (float(_kp[i].cmp_to_other(_kp[i + 1])), _kp[i], i + 1) for i in range(len(_kp) - 1)
        ]

        assert isinstance(flatten, list)
        flatten.sort(key=lambda tup: tup[0])

        mask = np.ones(_kp.shape, dtype=bool)
        prev_kp = flatten[0][1]
        for eq, k, p in flatten[1:]:
            mask[p] = (prev_kp != k)
            prev_kp = k

        self.__key_points = self.__key_points[mask]
        mask = tf.constant(mask, dtype=tf.bool)
        prev_values = self.__pack_backend()
        prev_values = tf.boolean_mask(prev_values, mask, axis=0)
        self.__unpack_backend(prev_values)

    def remove_keys(self, indices):
        if not isinstance(indices, (int, list, tuple)):
            raise ValueError('expected indices to be type of (int, list, tuple, np.ndarray, tf.Tensor)')
        _type = type(indices)
        tf_vals = self.__pack_backend()

        if _type == int:
            left, _, right = tf.split(tf_vals, [indices, 1, len(self) - indices - 1], axis=0)
            tf_vals = tf.concat((left, right), axis=0)
            self.__unpack_backend(tf_vals)
            left, _, right = np.hsplit(self.__key_points, [indices, indices + 1])
            self.__key_points = np.hstack((left, right))
        elif _type == list or _type == tuple:
            mask = [True] * len(self)
            prev_ind = 0
            indices.sort()

            for i, ind in enumerate(indices):
                if isinstance(ind, (list, tuple)): ind = ind[0]
                if not isinstance(ind, int): raise TypeError(f'expected type of int got {type(ind)}')
                if ind - prev_ind == 0 and ind != 0: continue
                mask[ind] = False
                prev_ind = ind

            tf_vals = tf.boolean_mask(tf_vals, tf.constant(mask, dtype=tf.bool))
            self.__unpack_backend(tf_vals)
            self.__key_points = self.__key_points[np.array(mask, dtype=bool)]

    def pack_values(self):
        values_order = ['pt', 'size', 'angle', 'octave', 'octave_id', 'response']
        return self.__pack_backend(), values_order

    def unpack_values(self, values):
        return self.__unpack_backend(values, set_backend=False)

    def as_size_image(self):
        if self.size_image:
            return None
        pt = self.pt * tf.constant([1.0, 0.5, 0.5, 1.0], dtype=self.pt.dtype)
        size = self.size * 0.5
        octave = bitwise_ops.bitwise_xor(self.octave, 255)
        new = type(self)(f'{self.name}_2', True, self.initial_size)
        new.add_keys(
            pt=pt, size=size, octave=octave, octave_id=tf.identity(self.octave_id),
            angle=tf.identity(self.angle), response=tf.identity(self.response)
        )
        return new

    class KeyPoint:
        def __init__(
                self,
                pt: tf.Tensor,
                size: tf.Tensor,
                angle: Union[tf.Tensor, float] = 0.0,
                octave: Union[tf.Tensor, int] = 0,
                octave_id: Union[tf.Tensor, int] = 0,
                response: Union[tf.Tensor, float] = -1.0,
                as_size_image: bool = False
        ):
            self.pt = tf.cast(pt, dtype=tf.float32)
            self.size = tf.cast(size, dtype=tf.float32)
            self.angle = tf.cast(angle, dtype=tf.float32)
            self.octave = tf.cast(octave, dtype=tf.int32)
            self.octave_id = tf.cast(octave_id, dtype=tf.int32)
            self.response = tf.cast(response, dtype=tf.float32)
            self.size_image = as_size_image

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                raise TypeError
            if tf.reduce_max(tf.abs(self.pt - other.pt)) != 0: return False
            if self.size != other.size: return False
            if self.angle != other.angle: return False
            if self.response != other.response: return False
            if self.octave != other.octave: return False
            if self.octave_id != other.octave_id: return False
            return True

        def cmp_to_other(self, other):
            if not isinstance(other, type(self)):
                raise ValueError
            p1 = tf.unstack(self.pt, num=4, axis=-1)
            p2 = tf.unstack(other.pt, num=4, axis=-1)

            if p1[0] != p2[0]: return p1[0] - p2[0]
            if p1[1] != p2[1]: return p1[1] - p2[1]
            if p1[2] != p2[2]: return p1[2] - p2[2]
            if p1[3] != p2[3]: return p1[3] - p2[3]
            if self.size != other.size: return self.size - other.size
            if self.angle != other.angle: return self.angle - other.angle
            if self.response != other.response: return self.response - other.response
            if self.octave != other.octave: return self.octave - other.octave
            return self.octave_id - other.octave_id

        @property
        def unpacked_octave(self):
            if not self.size_image:
                return None
            octave = bitwise_ops.bitwise_and(self.octave, 255)
            layer = bitwise_ops.right_shift(self.octave, 8) & 255
            if octave >= 128: octave = bitwise_ops.bitwise_or(octave, -128)
            scale = 1 / tf.bitwise.left_shift(1, octave) if octave >= 1 else tf.bitwise.left_shift(1, -octave)
            return tf.concat([tf.cast(a[..., tf.newaxis], dtype=tf.float32) for a in [octave, layer, scale]], axis=1)

        def as_size_image(self):
            pt = self.pt * tf.constant([1.0, 0.5, 0.5, 1.0], dtype=self.pt.dtype)
            size = self.size * 0.5
            octave = (self.octave & ~255) | ((self.octave - 1) & 255)
            return type(self)(pt, size, self.angle, octave, self.octave_id, self.response)

        def descriptors(self, gaussian_image: tf.Tensor, num_bins: int = 8, window_width: float = 4.0,
                        scale_multiplier: float = 3):
            # if not self.size_image:
            #     return None
            # _shape = tf.shape(gaussian_image)
            # up_octave = self.unpacked_octave
            # octave, layer, scale = tf.split(up_octave, [1, 1, 1], axis=1)
            # scale_pad = tf.pad(tf.repeat(scale, 2, axis=0), paddings=tf.constant([[0, 0], [1, 1]]), constant_values=1.0)
            # point = tf.cast(tf.round(self.pt * scale), dtype=tf.int64)
            # bins_per_degree = num_bins / 360.
            # angle = 360. - self.angle
            # cos_angle = math.cos((PI / 180) * angle)
            # sin_angle = math.sin((PI / 180) * angle)
            # weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            #
            # hist_width = scale_multiplier * 0.5 * tf.reshape(scale, (-1,)) * self.size
            #
            # hist_width = tf.cast(
            #     tf.round(
            #         hist_width * math.sqrt(2.0) * (window_width + 1.0) * 0.5
            #     ), dtype=tf.int32
            # )
            #
            # hist_width = math.minimum(hist_width, tf.reduce_max(_shape))
            # oc = alg.octave_pyramid[0]
            #
            # point, cos_angle, sin_angle, weight_multiplier, hist_width = k1.descriptors(oc.gaus_X)
            #
            # cords = make_neighborhood2D(point, con=int(hist_width * 2), origin_shape=oc.gaus_X.shape)
            #
            # gaus_img = tf.gather_nd(oc.gaus_X, tf.reshape(cords, shape=(-1, 4)))
            # gaus_img = tf.reshape(gaus_img, shape=(1, int(hist_width * 2), int(hist_width * 2), 1))
            #
            # gaus_grad = compute_central_gradient2D(gaus_img)
            #
            # dx, dy = tf.split(gaus_grad, [1, 1], axis=-1)
            # magnitude = tf.reshape(math.sqrt(dx * dx + dy * dy), (-1,))
            # orientation = tf.reshape((math.atan2(dx, dy) * (180.0 / PI)) % 360, (-1,))
            #
            # block_ = make_neighborhood2D(tf.constant([[0, 0, 0, 0]], dtype=tf.int64), con=int((hist_width - 1) * 2))
            # block_ = tf.cast(block_, dtype=tf.float32)
            # block_ = tf.reshape(block_, (-1, 4))
            # _, y, x, _ = tf.split(block_, [1, 1, 1, 1], axis=-1)
            #
            # y_rot = x * sin_angle + y * cos_angle
            # x_rot = x * cos_angle - y * sin_angle
            # y_bin = (y_rot / hist_width) + 0.5 * 4 - 0.5
            # x_bin = (x_rot / hist_width) + 0.5 * 4 - 0.5
            #
            # hist_width = tf.cast(hist_width, dtype=tf.float32)
            #
            # weight = math.exp(weight_multiplier * ((y_rot / hist_width) ** 2 + (x_rot / hist_width) ** 2))
            #
            # orientation_frac_ = orientation - math.floor(orientation)
            # orientation_frac_ = tf.reshape(orientation_frac_, (-1,))
            # y_bin_frac_ = y_bin - math.floor(y_bin)
            # x_bin_frac_ = x_bin - math.floor(x_bin)
            #
            # orientation_bin_floor = math.floor(orientation)
            # orientation_bin_floor_plus = orientation_bin_floor + 8
            # orientation_bin_floor = tf.where(
            #     math.logical_and(orientation_bin_floor < 0, orientation_bin_floor_plus < 8), orientation_bin_floor_plus,
            #     orientation_bin_floor)
            #
            # c1 = magnitude * y_bin_frac_
            # c0 = magnitude * (1 - y_bin_frac_)
            # c11 = c1 * x_bin_frac_
            # c10 = c1 * (1 - x_bin_frac_)
            # c01 = c0 * x_bin_frac_
            # c00 = c0 * (1 - x_bin_frac_)
            # c111 = c11 * orientation_frac_
            # c110 = c11 * (1 - orientation_frac_)
            # c101 = c10 * orientation_frac_
            # c100 = c10 * (1 - orientation_frac_)
            # c011 = c01 * orientation_frac_
            # c010 = c01 * (1 - orientation_frac_)
            # c001 = c00 * orientation_frac_
            # c000 = c00 * (1 - orientation_frac_)
            return


class SIFT:

    def __init__(
            self,
            sigma: float = 1.6,
            n_intervals: int = 3,
            name: Union[str, None] = None
    ):
        self.name = name or 'SIFT'
        self.__DTYPE = backend.floatx()
        self.__n_bins = 36
        self.__eigen_ration = 10
        self.__peak_ratio = 0.8
        self.__contrast_threshold = 0.04
        self.__scale_factor = 1.5
        self.__extrema_offset = 0.5
        self.__radius_factor = 3
        self.__con = 3
        self.epsilon = 1e-05
        self.octave_pyramid = []
        self.X_base = None
        self.gaussian_kernels = None
        self.n_octaves = None
        self.n_intervals = tf.cast(n_intervals, dtype=tf.int32)
        self.sigma = tf.cast(sigma, dtype=self.__DTYPE)
        self.key_points = KeyPointsSift()
        self.border_width = (3, 3, 0)
        self.__gauss_shapes = []

    def build_pyramid(
            self,
            inputs: tf.Tensor,
            sigma: float = 1.6,
            assume_blur_sigma: float = 0.5,
            num_octaves: Union[int, None] = None,
            num_intervals: int = 3
    ):
        _shape = tf.shape(inputs)
        _n_dims = len(_shape)
        if _n_dims != 4 or _shape[-1] != 1:
            raise ValueError(
                'expected the inputs to be grayscale images with size of (None, h, w, 1)'
            )
        self.key_points.initial_size = _shape
        b, h, w, d = tf.unstack(_shape, num=4, axis=-1)
        self.sigma = tf.cast(sigma, dtype=self.__DTYPE)
        self.n_intervals = tf.cast(num_intervals, dtype=self.__DTYPE)

        # calculate the base image with up sampling with factor 2
        self.X_base = inputs
        X_base = tf.image.resize(inputs, size=[h * 2, w * 2], method='bilinear', name='X_base')

        assume_blur = tf.cast(assume_blur_sigma, dtype=self.__DTYPE)
        delta_sigma = (self.sigma ** 2) - ((2 * assume_blur) ** 2)
        delta_sigma = math.sqrt(tf.maximum(delta_sigma, 0.64))

        X_base = gaussian_blur(X_base, kernel_size=0, sigma=delta_sigma)

        # generate the gaussian kernels

        images_per_octaves = self.n_intervals + 3
        K = 2 ** (1 / self.n_intervals)

        sigma_prev = (K ** (tf.cast(tf.range(1, images_per_octaves), dtype=self.__DTYPE) - 1.0)) * self.sigma
        sigmas = math.sqrt((K * sigma_prev) ** 2 - sigma_prev ** 2)

        self.gaussian_kernels = [
            gaussian_kernel(kernel_size=0, sigma=s) for s in sigmas
        ]

        # compute number of octaves
        min_shape = int(tf.shape(self.gaussian_kernels[-1])[0])
        max_n_octaves = self.compute_default_N_octaves(image_shape=tf.shape(X_base), min_shape=min_shape)
        if num_octaves is not None and max_n_octaves > num_octaves:
            max_n_octaves = tf.cast(num_octaves, dtype=tf.int32)
        self.n_octaves = max_n_octaves

        # building the scale pyramid
        def _conv(X, H):
            k_ = int(tf.shape(H)[0])
            paddings = tf.constant(
                [[0, 0], [k_ // 2, k_ // 2], [k_ // 2, k_ // 2], [0, 0]], dtype=tf.int32
            )
            X_pad = tf.pad(X, paddings, mode='SYMMETRIC')
            H = tf.reshape(H, shape=(k_, k_, 1, 1))
            return tf.nn.convolution(X_pad, H, padding='VALID')

        octave_base = tf.identity(X_base)

        for oc in tf.range(self.n_octaves):
            gaussian_X = [tf.identity(octave_base)]
            DOG_X = []
            for kernel in self.gaussian_kernels:
                prev_x = gaussian_X[-1]
                next_X = _conv(prev_x, kernel)
                gaussian_X.append(next_X)
                DOG_X.append(next_X - prev_x)

            _, Oh, Ow, _ = tf.unstack(tf.shape(octave_base), num=4)
            octave_base = tf.image.resize(gaussian_X[-3], size=[Oh // 2, Ow // 2], method='nearest')

            gaussian_X = tf.concat(gaussian_X, axis=-1)
            self.__gauss_shapes.append(tf.shape(gaussian_X))
            DOG_X = tf.concat(DOG_X, axis=-1)

            self.octave_pyramid.append(Octave(oc, gaussian_X, DOG_X))

    def extract_keypoints(
            self,
            n_iterations: Union[tf.Tensor, int] = 5,
            contrast_threshold: Union[tf.Tensor, float] = 0.04
    ) -> KeyPointsSift:

        temp_key_point = KeyPointsSift()

        n_iterations = int(n_iterations)
        threshold = tf.floor(0.5 * contrast_threshold / 3.0 * 255.0)
        border_width = self.border_width

        for octave in self.octave_pyramid:
            dog = octave.dog_X
            continue_search = compute_extrema3D(
                dog, threshold=threshold, con=self.__con, border_width=border_width, epsilon=self.epsilon
            )

            for _ in tf.range(n_iterations):
                continue_search, _ = self.__localize_extrema(
                    continue_search, dog, octave.octave_id, keyPoint_pointer=temp_key_point
                )

                if continue_search is None:
                    break

        histogram = tf.constant([[]], shape=(0, self.__n_bins), dtype=self.__DTYPE)

        del_points = []
        for k in range(len(temp_key_point)):
            curr_kp = temp_key_point[k]
            curr_hist = self.__compute_histogram(curr_kp)
            if curr_hist is None:
                del_points.append(k)
            else:
                histogram = tf.concat((histogram, tf.reshape(curr_hist, shape=(1, -1))), axis=0)

        temp_key_point.remove_keys(del_points)

        gaussian1D = tf.constant([1, 4, 6, 4, 1], dtype=self.__DTYPE) / 16.0
        gaussian1D = tf.reshape(gaussian1D, shape=(-1, 1, 1))

        histogram_pad = tf.pad(
            tf.expand_dims(histogram, axis=-1),
            paddings=tf.constant([[0, 0], [2, 2], [0, 0]], dtype=tf.int32),
            mode='SYMMETRIC'
        )

        smooth_histogram = tf.nn.convolution(histogram_pad, gaussian1D, padding='VALID')
        smooth_histogram = tf.squeeze(smooth_histogram, axis=-1)

        orientation_max = tf.reduce_max(smooth_histogram, axis=-1)
        value_cond = tf.repeat(tf.reshape(orientation_max, shape=(-1, 1)), repeats=self.__n_bins, axis=-1)
        value_cond = value_cond * self.__peak_ratio

        cond = math.logical_and(
            math.greater(smooth_histogram, tf.roll(smooth_histogram, shift=1, axis=1)),
            math.greater(smooth_histogram, tf.roll(smooth_histogram, shift=-1, axis=1))
        )
        cond = math.logical_and(
            cond, math.greater_equal(smooth_histogram, value_cond)
        )

        peak_index = tf.where(cond)
        peak_value = tf.boolean_mask(smooth_histogram, cond)

        p_id, p_idx = tf.unstack(peak_index, num=2, axis=-1)

        left_index = (p_idx - 1) % self.__n_bins
        right_index = (p_idx + 1) % self.__n_bins

        left_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, left_index), axis=-1))
        right_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, right_index), axis=-1))

        interpolated_peak_index = ((tf.cast(p_idx, dtype=self.__DTYPE) + 0.5 * (left_value - right_value)) / (
                left_value - (2 * peak_value) + right_value)) % self.__n_bins

        orientation = 360. - interpolated_peak_index * 360. / self.__n_bins

        orientation = tf.where(math.less(math.abs(orientation), 1e-7), 0.0, orientation)

        p_id = tf.reshape(p_id, (-1, 1))

        pack_keys, names = temp_key_point.pack_values()
        pack_keys = tf.gather_nd(pack_keys, p_id)
        unpack_keys = self.key_points.unpack_values(pack_keys)

        add_keys = {}
        for name, val in zip(names, unpack_keys):
            if name == 'angle':
                val = orientation
            add_keys[name] = val

        self.key_points.add_keys(**add_keys)
        return self.key_points

    def compute_default_N_octaves(
            self,
            image_shape: Union[tf.Tensor, list, tuple],
            min_shape: int = 0
    ) -> tf.Tensor:
        assert len(image_shape) == 4
        b, h, w, d = tf.unstack(image_shape, num=4)

        s_ = tf.cast(min([h, w]), dtype=self.__DTYPE)
        diff = math.log(s_)
        if min_shape > 1:
            diff = diff - math.log(tf.cast(min_shape, dtype=self.__DTYPE))

        n_octaves = tf.round(diff / math.log(2.0)) + 1
        return tf.cast(n_octaves, tf.int32)

    def __localize_extrema(
            self,
            middle_pixel_cords: tf.Tensor,
            dOg_array: tf.Tensor,
            octave_index: Union[int, float],
            keyPoint_pointer: Union[KeyPointsSift, None] = None
    ) -> tuple[Union[tf.Tensor, None], Union[tuple, None, KeyPointsSift]]:
        dog_shape_ = dOg_array.shape

        cube_neighbor = make_neighborhood3D(middle_pixel_cords, con=self.__con, origin_shape=dog_shape_)
        if cube_neighbor.shape[0] == 0:
            return None, None

        # the size can change because the make_neighborhood3D return only the valid indexes
        middle_pixel_cords = cube_neighbor[:, (self.__con ** 3) // 2, :]

        cube_values = tf.gather_nd(dOg_array, tf.reshape(cube_neighbor, shape=(-1, 4))) / 255.0
        cube_values = tf.reshape(cube_values, (-1, self.__con, self.__con, self.__con))

        grad = compute_central_gradient3D(cube_values)
        grad = tf.reshape(grad, shape=(-1, 3, 1))
        hess = compute_hessian_3D(cube_values)
        hess = tf.reshape(hess, shape=(-1, 3, 3))
        extrema_update = - linalg.lstsq(hess, grad, l2_regularizer=0.0, fast=False)
        extrema_update = tf.squeeze(extrema_update, axis=-1)

        next_step_cords, kp_cond_1 = self.__split_extrema_cond(extrema_update, middle_pixel_cords)

        dot_ = tf.reduce_sum(
            tf.multiply(
                extrema_update[:, tf.newaxis, tf.newaxis, ...],
                tf.transpose(grad, perm=(0, 2, 1))[:, tf.newaxis, ...]
            ), axis=-1, keepdims=False
        )
        dot_ = tf.reshape(dot_, shape=(-1,))
        update_response = cube_values[:, 1, 1, 1] + 0.5 * dot_

        kp_cond_2 = math.greater_equal(
            math.abs(update_response) * int(self.n_intervals), self.__contrast_threshold - self.epsilon
        )

        hess_xy = hess[:, :2, :2]
        hess_xy_trace = linalg.trace(hess_xy)
        hess_xy_det = linalg.det(hess_xy)

        kp_cond_3 = math.logical_and(
            math.greater(hess_xy_det, 0.0),
            math.less(self.__eigen_ration * (hess_xy_trace ** 2), ((self.__eigen_ration + 1) ** 2) * hess_xy_det)
        )
        sure_key_points = math.logical_and(math.logical_and(kp_cond_1, kp_cond_2), kp_cond_3)

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
        kp_octave = tf.cast(kp_octave, dtype=tf.int32)

        kp_size = self.sigma * (2 ** ((cz + ez) / tf.cast(self.n_intervals, dtype=self.__DTYPE))) * (
                2 ** (octave_index + 1.0))
        kp_response = math.abs(tf.boolean_mask(update_response, sure_key_points))

        if keyPoint_pointer is not None:
            keyPoint_pointer.add_keys(pt=kp_pt, size=kp_size, octave=kp_octave,
                                      octave_id=tf.cast(octave_index, dtype=tf.int32), response=kp_response)
            return next_step_cords, keyPoint_pointer
        kp = namedtuple('KeyPoint', 'pt, octave, o_size, response, octave_index')
        kp = kp(kp_pt, kp_octave, kp_size, kp_response, tf.cast(octave_index, dtype=tf.int32))
        return next_step_cords, kp

    def __split_extrema_cond(
            self,
            extrema: tf.Tensor,
            current_cords: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        shape_ = extrema.get_shape()
        assert shape_[-1] == 3
        assert len(shape_) == 2

        cond = math.less(math.reduce_max(math.abs(extrema), axis=-1), self.__extrema_offset)

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

        check_if_change = tf.reduce_max(math.abs(next_step_cords_temp - next_step_cords), axis=-1)

        next_step_cords = tf.boolean_mask(next_step_cords, math.greater(check_if_change, 0))

        return next_step_cords, cond

    def __compute_histogram(
            self,
            key_point: KeyPointsSift.KeyPoint
    ) -> tf.Tensor:
        octave = self.octave_pyramid[int(key_point.octave_id)]
        scale = self.__scale_factor * key_point.size / (2 ** (tf.cast(key_point.octave_id, dtype=self.__DTYPE) + 1))
        radius = tf.cast(tf.round(self.__radius_factor * scale), dtype=tf.int32)
        weight_factor = -0.5 / (scale ** 2)

        _prob = 1.0 / tf.cast(2 ** key_point.octave_id, dtype=self.__DTYPE)
        _one = tf.ones_like(_prob)
        _prob = tf.stack((_one, _prob, _prob, _one), axis=-1)

        region_center = tf.cast(key_point.pt * _prob, dtype=tf.int64)

        con = int((radius * 2) + 3)
        block = make_neighborhood2D(region_center, con=con, origin_shape=octave.gaus_X.shape)

        if block.shape[0] == 0:
            return None

        gaus_img = tf.gather_nd(octave.gaus_X, tf.reshape(block, shape=(-1, 4)))
        gaus_img = tf.reshape(gaus_img, shape=(1, con, con, 1))
        gaus_grad = compute_central_gradient2D(gaus_img)

        dx, dy = tf.split(gaus_grad, [1, 1], axis=-1)
        magnitude = math.sqrt(dx * dx + dy * dy)
        orientation = math.atan2(dx, dy) * (180.0 / PI)

        neighbor_index = make_neighborhood2D(tf.constant([[0, 0, 0, 0]], dtype=tf.int64), con=con - 2)
        neighbor_index = tf.cast(tf.reshape(neighbor_index, shape=(1, con - 2, con - 2, 4)), dtype=self.__DTYPE)
        _, y, x, _ = tf.split(neighbor_index, [1, 1, 1, 1], axis=-1)

        weight = math.exp(weight_factor * (y ** 2 + x ** 2))
        hist_index = tf.cast(tf.round(orientation * self.__n_bins / 360.), dtype=tf.int64)
        hist_index = hist_index % self.__n_bins
        hist_index = tf.reshape(hist_index, (-1, 1))

        hist = tf.zeros(shape=(self.__n_bins,), dtype=self.__DTYPE)
        hist = tf.tensor_scatter_nd_add(hist, hist_index, tf.reshape(weight * magnitude, (-1,)))
        return hist


if __name__ == '__main__':
    img = tf.keras.utils.load_img('box.png', color_mode='grayscale')
    img = tf.convert_to_tensor(tf.keras.utils.img_to_array(img), dtype=tf.float32)
    img = img[tf.newaxis, ...]

    alg = SIFT(sigma=1.6)
    alg.build_pyramid(img, num_octaves=4, num_intervals=5)
    # kp = alg.extract_keypoints()
    # kp.remove_duplicate()
    #
    # K = kp.as_size_image()
    # k1 = K[0]

