import tensorflow as tf
from tensorflow import keras

PI = tf.cast(
    tf.math.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32
)


class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 learning_rate_base,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 name=None):

        super(WarmUpCosineDecay, self).__init__()
        if (total_steps is None and epochs is None and steps_per_epoch is None) or (
                total_steps is None and (epochs is None or steps_per_epoch is None)
        ):
            raise ValueError(
                'Missing parameter for steps.'
                ' You must provide either the total_steps parameter or both the epochs and steps_per_epoch parameters'
            )
        self.total_steps = total_steps if total_steps is not None else epochs * steps_per_epoch

        if self.total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to warmup_steps.')

        if (warmup_steps > 0) and (learning_rate_base < warmup_learning_rate):
            raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')

        self.learning_rate_base = learning_rate_base
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.name = name

    def __call__(self, step):
        lr_base = tf.convert_to_tensor(self.learning_rate_base, name='init_lr')
        _type = lr_base.dtype

        total_steps = tf.cast(self.total_steps, dtype=_type, name='total_steps')
        warmup_lr = tf.cast(self.warmup_learning_rate, dtype=_type, name='warmup_lr')
        warmup_steps = tf.cast(self.warmup_steps, dtype=_type, name='warmup_steps')
        hold_steps = tf.cast(self.hold_base_rate_steps, dtype=_type, name='hold_steps')

        global_step = tf.cast(step, dtype=_type, name='global_step')
        pi = tf.cast(PI, dtype=_type)

        with tf.name_scope(self.name or "WarmUpCosineDecay"):
            slope = (lr_base - warmup_lr) / warmup_steps

            lr = 0.5 * lr_base * (1 + tf.math.cos(pi * (
                    (global_step - warmup_steps - hold_steps) / (total_steps - warmup_steps - hold_steps)
            )))

            if self.hold_base_rate_steps > 0:
                lr = tf.where(
                    tf.greater(global_step, warmup_steps + hold_steps), lr, lr_base
                )

            if self.warmup_steps > 0:
                warmup_rate = slope * global_step + warmup_lr
                lr = tf.where(
                    tf.greater(warmup_steps, global_step), warmup_rate, lr
                )

            lr = tf.where(
                tf.greater(global_step, total_steps), 0.0, lr
            )

        return lr

    def get_config(self):
        return dict(
            total_steps=self.total_steps,
            learning_rate_base=self.learning_rate_base,
            warmup_learning_rate=self.warmup_learning_rate,
            warmup_steps=self.warmup_steps,
            hold_base_rate_steps=self.hold_base_rate_steps,
            name=self.name
        )
