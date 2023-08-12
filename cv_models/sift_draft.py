from typing import Union
import tensorflow as tf
from tensorflow.python.keras import backend
import numpy as np

math = tf.math
linalg = tf.linalg

backend.set_floatx('float32')


# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf


class KeyPointsSift:
    __DTYPE = backend.floatx()

    def __init__(self, name=None):
        self.name = name or 'KeyPointsSift'
        self.pt = tf.constant([[]], shape=(0, 4), dtype=self.__DTYPE)
        self.size = tf.constant([[]], shape=(0,), dtype=self.__DTYPE)
        self.angle = tf.constant([[]], shape=(0,), dtype=self.__DTYPE)
        self.octave = tf.constant([[]], shape=(0,), dtype=tf.int32)
        self.octave_id = tf.constant([[]], shape=(0,), dtype=tf.int32)
        self.response = tf.constant([[]], shape=(0,), dtype=self.__DTYPE)
        self.__key_points = np.array([], dtype=object)
        self.__key_built = False

    def __len__(self):
        return int(self.pt.get_shape()[0])

    def __iter__(self):
        self.__i = 0
        return self

    def __next__(self):
        if self.__i == len(self):
            del self.__i
            raise StopIteration
        self.__i += 1
        return self.key_points[self.__i - 1]

    def __getitem__(self, index):
        if not isinstance(index, (int, slice)) or not (
                isinstance(index, (list, tuple)) and isinstance(index[0], (list, tuple))):
            raise KeyError(f'{index}')
        return self.key_points[index]

    def __delitem__(self, index):
        if not isinstance(index, int):
            raise KeyError(f'{index}')

        prev_len = len(self)
        prev_values = self.__backend
        ret_values = []

        for val in prev_values:
            left, _, right = tf.split(val, [index, 1, prev_len - index - 1], axis=0)
            ret_values.append(tf.concat((left, right), axis=0))
        self.__backend = ret_values

        if self.__key_built:
            left, _, right = np.hsplit(self.__key_points, [index, index + 1])
            self.__key_points = np.hstack((left, right))

    def __remove_with_mask(self, mask):
        if isinstance(mask, (list, tuple)):
            bool_ = isinstance(mask[0], (list, tuple))
            for ind in mask:
                ind = ind[0] if bool_ else ind
                assert isinstance(ind, int)
                del self[ind]
            return
        if not isinstance(mask, np.ndarray) and not isinstance(mask.dtype, bool):
            raise ValueError
        self.__key_points = self.__key_points[mask]
        indices = tf.convert_to_tensor(mask, dtype=tf.bool)
        prev_values = self.__backend
        ret_values = []

        for val in prev_values:
            ret_values.append(tf.boolean_mask(val, indices, axis=0))

        self.__backend = ret_values

    @property
    def __backend(self):
        arrays = [
            self.pt, self.size, self.angle, self.octave, self.octave_id, self.response
        ]
        return arrays

    @__backend.setter
    def __backend(self, values):
        lab = ['pt', 'size', 'angle', 'octave', 'octave_id', 'response']
        for key, val in zip(lab, values):
            self.__setattr__(key, val)
        return

    @property
    def key_points(self):
        _len = len(self)
        _keys_len = len(self.__key_points)
        if _len == _keys_len:
            return self.__key_points
        self.__key_built = True
        diff = _len - _keys_len
        split_index = [_keys_len] + [1] * diff
        backend_arr = self.__backend
        splits = [
            tf.split(arr, split_index, axis=0)[1:] for arr in backend_arr
        ]

        for index, split in enumerate(zip(*splits)):
            key = self.KeyPoint(
                pt=split[0], size=split[1], angle=split[2],
                octave=split[3], octave_id=split[4], response=split[5]
            )
            self.__key_points = np.append(self.__key_points, key)

        return self.__key_points

    def add_keys(self,
                 pt: Union[tf.Tensor, list, tuple],
                 size: Union[tf.Tensor, float],
                 angle: Union[tf.Tensor, float] = 0.0,
                 octave: Union[tf.Tensor, int] = 0,
                 octave_id: Union[tf.Tensor, int] = 0,
                 response: Union[tf.Tensor, float] = -1.0):

        size = tf.cast(tf.reshape(size, shape=(-1,)), dtype=self.__DTYPE)
        n_points_ = int(tf.shape(size)[0])
        self.size = tf.concat((self.size, size), axis=0)

        pt = tf.cast(tf.reshape(pt, shape=(-1, 4)), dtype=self.__DTYPE)
        assert pt.shape[0] == n_points_
        self.pt = tf.concat((self.pt, pt), axis=0)

        def map_args(arg):
            if not isinstance(arg, tf.Tensor):
                return tf.constant([arg] * n_points_)
            arg = tf.reshape(arg, shape=(-1,))
            shape_ = tf.shape(arg)[0]
            if shape_ == 1 and n_points_ > 1:
                return tf.repeat(arg, repeats=n_points_ - 1, axis=0)
            assert arg.shape[0] == n_points_
            return arg

        args = [angle, octave, octave_id, response]
        angle, octave, octave_id, response = list(map(map_args, args))

        angle = tf.cast(angle, dtype=self.__DTYPE)
        self.angle = tf.concat((self.angle, angle), axis=0)

        octave = tf.cast(octave, dtype=tf.int32)
        self.octave = tf.concat((self.octave, octave), axis=0)

        octave_id = tf.cast(octave_id, dtype=tf.int32)
        self.octave_id = tf.concat((self.octave_id, octave_id), axis=0)

        response = tf.cast(response, dtype=self.__DTYPE)
        self.response = tf.concat((self.response, response), axis=0)

    def remove_duplicate(self):
        def compare(keypoint1, keypoint2):
            p1 = tf.unstack(keypoint1.pt, num=4, axis=-1)
            p2 = tf.unstack(keypoint2.pt, num=4, axis=-1)

            if p1[0] != p2[0]:
                return float(p1[0] - p2[0])

            if p1[1] != p2[1]:
                return float(p1[1] - p2[1])

            if p1[2] != p2[2]:
                return float(p1[2] - p2[2])

            if p1[3] != p2[3]:
                return float(p1[3] - p2[3])

            if keypoint1.size != keypoint2.size:
                return float(keypoint2.size - keypoint1.size)

            if keypoint1.angle != keypoint2.angle:
                return float(keypoint1.angle - keypoint2.angle)

            if keypoint1.response != keypoint2.response:
                return float(keypoint2.response - keypoint1.response)

            if keypoint1.octave != keypoint2.octave:
                return float(keypoint2.octave - keypoint1.octave)

            return float(keypoint2.octave_id - keypoint1.octave_id)

        _kp = self.key_points
        flatten = [
            (compare(_kp[i], _kp[i + 1]), _kp[i], i + 1) for i in range(len(_kp) - 1)
        ]

        assert isinstance(flatten, list)
        flatten.sort(key=lambda tup: tup[0])

        indices = np.ones(_kp.shape, dtype=bool)
        prev_kp = flatten[0][1]
        for eq, k, p in flatten[1:]:
            indices[p] = (prev_kp != k)
            prev_kp = k

        self.__remove_with_mask(indices)

    class KeyPoint:
        def __init__(self, pt, size, angle=0.0, octave=0, octave_id=0, response=-1.0):
            self.pt = tf.cast(pt, dtype=tf.float32)
            self.size = tf.cast(size, dtype=tf.float32)
            self.angle = tf.cast(angle, dtype=tf.float32)
            self.octave = tf.cast(octave, dtype=tf.int32)
            self.octave_id = tf.cast(octave_id, dtype=tf.int32)
            self.response = tf.cast(response, dtype=tf.float32)

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                raise TypeError
            if tf.reduce_max(tf.abs(self.pt - other.pt)) != 0:
                return False
            if self.size != other.size:
                return False
            if self.angle != other.angle:
                return False
            if self.response != other.response:
                return False
            if self.octave != other.octave:
                return False
            if self.octave_id != other.octave_id:
                return False
            return True


if __name__ == '__main__':
    key_points = KeyPointsSift()

    temp = dict(
        _pt=[[0.0, 23.310443878173828, 512.0009765625, 1.0], [0.0, 45.54006576538086, 331.4445495605469, 3.0],
             [0.0, 80.75707244873047, 419.6999816894531, 1.0], [0.0, 80.75707244873047, 419.6999816894531, 1.0],
             [0.0, 82.90933990478516, 337.01361083984375, 1.0], [0.0, 90.93933868408203, 471.9446716308594, 1.0],
             [0.0, 110.0242919921875, 198.7282257080078, 3.0], [0.0, 110.0242919921875, 198.7282257080078, 3.0],
             [0.0, 117.1764144897461, 342.6973571777344, 1.0], [0.0, 119.52792358398438, 471.5077819824219, 3.0],
             [0.0, 143.0499725341797, 144.2001495361328, 1.0], [0.0, 164.12734985351562, 351.6940612792969, 3.0],
             [0.0, 175.1547393798828, 27.316211700439453, 1.0], [0.0, 189.41064453125, 200.00723266601562, 1.0],
             [0.0, 205.6959686279297, 559.353759765625, 4.0], [0.0, 212.86929321289062, 518.6109008789062, 3.0],
             [0.0, 217.40126037597656, 24.25643539428711, 1.0], [0.0, 217.7612762451172, 77.62590026855469, 3.0],
             [0.0, 218.06088256835938, 554.7293701171875, 1.0], [0.0, 238.63308715820312, 420.1122741699219, 1.0],
             [0.0, 243.49087524414062, 546.22509765625, 1.0], [0.0, 245.0523223876953, 273.9361572265625, 1.0],
             [0.0, 245.69728088378906, 47.339210510253906, 4.0], [0.0, 252.22630310058594, 266.43438720703125, 1.0],
             [0.0, 279.7511901855469, 55.416404724121094, 1.0], [0.0, 279.7511901855469, 55.416404724121094, 1.0],
             [0.0, 280.8144836425781, 43.910030364990234, 1.0], [0.0, 281.04095458984375, 335.0519104003906, 1.0],
             [0.0, 306.31658935546875, 56.99310302734375, 1.0], [0.0, 320.66156005859375, 427.1969299316406, 4.0],
             [0.0, 330.533935546875, 509.02850341796875, 4.0], [0.0, 338.5062561035156, 405.3922119140625, 4.0],
             [0.0, 349.65960693359375, 363.1810607910156, 3.0], [0.0, 360.0093688964844, 500.0473327636719, 1.0],
             [0.0, 360.0093688964844, 500.0473327636719, 1.0], [0.0, 361.84228515625, 382.84466552734375, 1.0],
             [0.0, 361.84228515625, 382.84466552734375, 1.0], [0.0, 379.924072265625, 469.5652160644531, 1.0],
             [0.0, 397.3897705078125, 494.9901123046875, 1.0], [0.0, 398.98382568359375, 384.7360534667969, 3.0],
             [0.0, 399.5465393066406, 508.3434753417969, 3.0], [0.0, 425.1154479980469, 505.291015625, 1.0],
             [0.0, 66.61227416992188, 507.83905029296875, 2.0], [0.0, 66.61227416992188, 507.83905029296875, 2.0],
             [0.0, 206.65914916992188, 468.8001708984375, 2.0], [0.0, 76.67898559570312, 615.5833740234375, 4.0],
             [0.0, 105.91239929199219, 369.1412048339844, 4.0], [0.0, 176.973876953125, 603.9937133789062, 4.0],
             [0.0, 187.4148712158203, 558.3486328125, 1.0], [0.0, 191.79783630371094, 176.4373016357422, 4.0],
             [0.0, 220.91062927246094, 22.09457778930664, 1.0], [0.0, 220.91062927246094, 22.09457778930664, 1.0],
             [0.0, 242.66775512695312, 557.16748046875, 1.0], [0.0, 246.15667724609375, 298.788818359375, 3.0],
             [0.0, 248.55343627929688, 172.8282012939453, 1.0], [0.0, 259.6634826660156, 311.34490966796875, 1.0],
             [0.0, 262.3655090332031, 70.21871185302734, 1.0], [0.0, 272.9185791015625, 203.00924682617188, 1.0],
             [0.0, 286.5994567871094, 548.6842651367188, 1.0], [0.0, 287.21966552734375, 85.222412109375, 1.0],
             [0.0, 328.2416687011719, 555.150634765625, 3.0], [0.0, 397.3768615722656, 284.5030517578125, 4.0],
             [0.0, 232.9132080078125, 550.8714599609375, 2.0], [0.0, 75.18125915527344, 488.1159362792969, 4.0],
             [0.0, 143.04940795898438, 51.16015625, 3.0], [0.0, 149.79835510253906, 50.300174713134766, 1.0],
             [0.0, 219.5962371826172, 268.36248779296875, 1.0]],
        _response=[0.010574450716376305, 0.021945638582110405, 0.012591552920639515, 0.012591552920639515,
                   0.02178187295794487, 0.014185680076479912, 0.03807948902249336, 0.03807948902249336,
                   0.010349166579544544, 0.010870518162846565, 0.023810364305973053, 0.010133933275938034,
                   0.012326379306614399, 0.01351283211261034, 0.02298770472407341, 0.009298657067120075,
                   0.022371578961610794, 0.017214719206094742, 0.015044737607240677, 0.0287704449146986,
                   0.009990496560931206, 0.037730395793914795, 0.020665863528847694, 0.029285583645105362,
                   0.04884669557213783, 0.04884669557213783, 0.010005129501223564, 0.016052814200520515,
                   0.04088236019015312, 0.01860039122402668, 0.015954071655869484, 0.008801442570984364,
                   0.010046242736279964, 0.027223294600844383, 0.027223294600844383, 0.008311967365443707,
                   0.008311967365443707, 0.011792846024036407, 0.008814462460577488, 0.01616630144417286,
                   0.01976492628455162, 0.01461009681224823, 0.014753773808479309, 0.014753773808479309,
                   0.022439679130911827, 0.009676499292254448, 0.03071306087076664, 0.018332181498408318,
                   0.019904600456357002, 0.013729233294725418, 0.02421514503657818, 0.02421514503657818,
                   0.02726774476468563, 0.018713176250457764, 0.018126657232642174, 0.03878019005060196,
                   0.03710372745990753, 0.013961107470095158, 0.008310848847031593, 0.014611709862947464,
                   0.018665574491024017, 0.03661053255200386, 0.0183781199157238, 0.025229984894394875,
                   0.0156688392162323, 0.018273893743753433, 0.01596587896347046],
        _octave=[8847616, 5440256, 3145984, 3145984, 5046528, 15008000, 8389376, 8389376, 9306368, 12845824, 12517632,
                 7013120, 10354944, 11337984, 16712704, 6947584, 5112064, 5243648, 917760, 9175296, 7078144, 14483712,
                 1442816, 2556160, 7602432, 7602432, 15925504, 5701888, 8257792, 1770496, 6685696, 1967104, 9110272,
                 11665664, 11665664, 256, 256, 4260096, 14024960, 5767936, 10224384, 14811392, 15729152, 15729152,
                 6554112, 3539969, 2491393, 11142145, 13500673, 8586241, 7864577, 7864577, 11206913, 4326145, 15663361,
                 7667969, 7012609, 12189953, 8782081, 2883841, 12845825, 11404289, 13238785, 15860738, 11272962,
                 13631746,
                 1573122],
        _angle=[3.45855712890625, 358.9222106933594, 3.195220947265625, 15.345489501953125, 10.943115234375,
                7.87750244140625, 2.418182373046875, 1.9932861328125, 23.305389404296875, 2.866668701171875,
                358.88824462890625, 356.5339050292969, 5.090301513671875, 5.35595703125, 358.45452880859375,
                2.9052734375,
                2.30767822265625, 3.1903076171875, 24.7784423828125, 359.7846374511719, 25.24273681640625,
                0.7349853515625, 358.611328125, 359.9819641113281, 0.4271240234375, 40.157867431640625,
                358.6650695800781,
                9.3961181640625, 357.9912414550781, 358.3415222167969, 5.375030517578125, 22.89727783203125,
                6.73736572265625, 5.50872802734375, 11.44134521484375, 95.989990234375, 198.0013427734375,
                61.6795654296875, 23.721099853515625, 3.139739990234375, 4.10992431640625, 1.2440185546875,
                0.983856201171875, 8.5084228515625, 359.119140625, 4.895111083984375, 2.342681884765625,
                3.138214111328125, 358.0776672363281, 356.93170166015625, 0.65130615234375, 1.7359619140625,
                358.43975830078125, 357.9928283691406, 3.253204345703125, 0.7777099609375, 10.022613525390625,
                9.328155517578125, 1.090240478515625, 358.2351989746094, 359.7026672363281, 3.043792724609375,
                358.1306457519531, 10.416107177734375, 359.4151916503906, 358.22906494140625, 11.75341796875],
        _octave_id=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        _size=[3.691697359085083, 4.734517574310303, 3.520314931869507, 3.520314931869507, 3.5763981342315674,
               3.884813070297241, 4.850391864776611, 4.850391864776611, 3.704204559326172, 5.034351825714111,
               3.805957794189453, 4.795794486999512, 3.738129138946533, 3.767005205154419, 5.97028112411499,
               4.793357849121094, 3.5779061317443848, 4.7277069091796875, 3.4556050300598145, 3.7010555267333984,
               3.637700319290161, 3.8667447566986084, 5.2610907554626465, 3.5037381649017334, 3.653883695602417,
               3.653883695602417, 3.91499400138855, 3.595534563064575, 3.672640562057495, 5.276390552520752,
               5.495144367218018, 5.283191680908203, 4.881593227386475, 3.778888463973999, 3.778888463973999,
               3.4297776222229004, 3.4297776222229004, 3.553253173828125, 3.852492094039917, 4.7478461265563965,
               4.927300453186035, 3.8784751892089844, 4.48945951461792, 4.48945951461792, 4.1603007316589355,
               10.709149360656738, 10.616381645202637, 11.4047212600708, 7.67333459854126, 11.16169548034668,
               7.323458194732666, 7.323458194732666, 7.52599573135376, 9.38366985321045, 7.812782287597656,
               7.309082984924316, 7.271755218505859, 7.58883810043335, 7.376185894012451, 7.023697853088379,
               10.069777488708496, 11.430525779724121, 8.793760299682617, 23.7208309173584, 19.879638671875,
               15.359094619750977, 13.898155212402344]
    )

    key_points.add_keys(pt=tf.convert_to_tensor(temp['_pt'][:25]),
                        size=tf.convert_to_tensor(temp['_size'][:25]),
                        angle=tf.convert_to_tensor(temp['_angle'][:25]),
                        octave=tf.convert_to_tensor(temp['_octave'][:25]),
                        octave_id=tf.convert_to_tensor(temp['_octave_id'][:25]),
                        response=tf.convert_to_tensor(temp['_response'][:25]))
    assert len(key_points.key_points) == 25
    key_points.add_keys(pt=tf.convert_to_tensor(temp['_pt'][20:]),
                        size=tf.convert_to_tensor(temp['_size'][20:]),
                        angle=tf.convert_to_tensor(temp['_angle'][20:]),
                        octave=tf.convert_to_tensor(temp['_octave'][20:]),
                        octave_id=tf.convert_to_tensor(temp['_octave_id'][20:]),
                        response=tf.convert_to_tensor(temp['_response'][20:]))
    assert len(key_points.key_points) == 72

    key_points.remove_duplicate()
    # assert len(key_points.key_points) == 67
    kp = key_points.key_points
    #
    # kp_change = kp[25]
    # assert kp_change.pointer is None
