from lr_schedules.schedules import WarmUpCosineDecay
from viz import learning_rate_viz


# learning_rate_viz(
#     WarmUpCosineDecay(
#         learning_rate_base=1e-2,
#         epochs=50,
#         steps_per_epoch=32,
#         warmup_steps=50,
#         hold_base_rate_steps=100
#     ), data_size=1000, epochs=50
# )


