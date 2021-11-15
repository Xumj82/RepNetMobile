from model.layers import SelfSimilarity
import tensorflow as tf
from model.PeriodEstimator import rep_net_mobile,PeriodEstimator

model = rep_net_mobile(input_shape=(2, 64,200,200,3))
model.summary()

# x = tf.random.normal(shape=(2, 64, 3, 3, 1280))
# y =  SelfSimilarity()(x)

# model = PeriodEstimator()
# model.build(input_shape=(2, 64,112,112,3))
# model.summary()