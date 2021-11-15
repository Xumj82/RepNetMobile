
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from .PeriodicityAccuracy import PeriodicityAccuracy
from .layers import feature_encoder, reduce_temporal_feature, self_similarity, period_embedding, periodicity_classifier, with_in_period_classifier
from .layers import FeatureEncoder,ReduceTemporalFeature,SelfSimilarity,PeriodEmbedding,PeriodicityClassifier,WithInPeriodClassifier

loss_tracker = tf.keras.metrics.Mean(name='loss')
loss_tracker1 = tf.keras.metrics.Mean(name='periodicity_loss')
loss_tracker2 = tf.keras.metrics.Mean(name='with_in_periody_loss')
metric_tracker1 = PeriodicityAccuracy(name='periodicity_acc', num_classes=32)
metric_tracker2 = tf.keras.metrics.BinaryAccuracy(name='with_in_period_acc')
metric_tracker3 = tf.keras.metrics.CosineSimilarity(name='periodicity_acc')
cce = tf.keras.losses.CategoricalCrossentropy()
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mae = tf.keras.losses.MeanAbsoluteError()

def rep_net_mobile(input_shape, ) -> Model:
    input = tf.keras.Input(input_shape[1:], input_shape[0], name="video_input")
    y = feature_encoder(input)(input) # (batch*frames, h, w, c1)
    y = reduce_temporal_feature(y)(y) # (batch, frames, c2)
    sim = y = self_similarity(y)(y) # (batch, frames, frames, 1)
    y = period_embedding(y)(y) # (batch, frames, embeddings)
    periodicity = periodicity_classifier(y)(y) # (batch, frames, frames//2)
    with_in_period = with_in_period_classifier(y)(y) # (batch, frames, 1)
    model = PeriodEstimator2([input], [periodicity, with_in_period, sim],name='period_estimator')
    return model

def rep_net_mobile_sim(input_shape,) -> Model:
    input = tf.keras.Input(input_shape[1:], input_shape[0], name="video_input")
    y = feature_encoder(input)(input) # (batch*frames, h, w, c1)
    y = reduce_temporal_feature(y)(y) # (batch, frames, c2)
    y = self_similarity(y)(y) # (batch, frames, frames, 1)
    model = Model([input], [y],name='period_estimator')
    return model

class PeriodEstimator(keras.Model):
    def __init__(self, ):
        super(PeriodEstimator, self).__init__()
        self.feature_encoder = FeatureEncoder()
        self.reduce_temporal_feature = ReduceTemporalFeature()
        self.self_similarity = SelfSimilarity()
        self.period_embedding = PeriodEmbedding()
        self.periodicity_classifier = PeriodicityClassifier()
        self.with_in_period_classifier = WithInPeriodClassifier()

    def call(self, inputs):
        x = self.feature_encoder(inputs)
        x = self.reduce_temporal_feature(x)
        x = self.self_similarity(x)
        x = self.period_embedding(x)
        y1 = self.periodicity_classifier(x)
        y2 = self.with_in_period_classifier(x)
        return y1, y2

    def train_step(self, data):
        x, y_true1, y_true2 = data

        with tf.GradientTape() as tape:
            periodicity, with_in_period = self(x)
            loss1 = cce(y_true1, periodicity)
            loss2 = bce(y_true2, with_in_period)
            loss = loss1+loss2
        trainable_vars = self.trainable_weights
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        

        loss_tracker.update_state(loss)
        loss_tracker1.update_state(loss1)
        loss_tracker2.update_state(loss2)
        metric_tracker1.update_state(y_true1, periodicity)
        metric_tracker2.update_state(y_true2, with_in_period)
        return {'loss': loss_tracker.result(),
                'periodicity_loss': loss_tracker1.result(),
                'with_in_periody_loss': loss_tracker2.result(),
                'periodicity_acc': metric_tracker1.result(),
                'with_in_period_acc': metric_tracker2.result()}

    def test_step(self, data):
        x, y_true1, y_true2 = data
        periodicity, with_in_period = self(x, training=False)
        y_true1 = tf.cast(tf.argmax(y_true1, axis=-1),dtype=tf.float32)
        periodicity =  tf.cast(tf.argmax(periodicity, axis=-1),dtype=tf.float32)
        loss1 = mae( y_true1, periodicity)
        loss2 = bce(y_true2, with_in_period)
        loss = loss1 + loss2
        loss_tracker.update_state(loss)
        loss_tracker1.update_state(loss1)
        loss_tracker2.update_state(loss2)
        metric_tracker3.update_state(y_true1,periodicity)
        metric_tracker2.update_state(y_true2, with_in_period)
        return {'loss': loss_tracker.result(),
                'periodicity_loss': loss_tracker1.result(),
                'with_in_periody_loss': loss_tracker2.result(),
                'periodicity_acc': metric_tracker3.result(),
                'with_in_period_acc': metric_tracker2.result()}
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker,loss_tracker1,loss_tracker2,metric_tracker3,metric_tracker2]

class PeriodEstimator2(keras.Model):
    def train_step(self, data):
        x, y_true1, y_true2 = data

        with tf.GradientTape() as tape:
            periodicity, with_in_period = self(x)
            loss1 = cce(y_true1, periodicity)
            loss2 = bce(y_true2, with_in_period)
            loss = loss1+loss2
        trainable_vars = self.trainable_weights
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        

        loss_tracker.update_state(loss)
        loss_tracker1.update_state(loss1)
        loss_tracker2.update_state(loss2)
        metric_tracker1.update_state(y_true1, periodicity)
        metric_tracker2.update_state(y_true2, with_in_period)
        return {'loss': loss_tracker.result(),
                'periodicity_loss': loss_tracker1.result(),
                'with_in_periody_loss': loss_tracker2.result(),
                'periodicity_acc': metric_tracker1.result(),
                'with_in_period_acc': metric_tracker2.result()}
    def test_step(self, data):
        x, y_true1, y_true2 = data
        periodicity, with_in_period = self(x, training=False)
        y_true1 = tf.cast(tf.argmax(y_true1, axis=-1),dtype=tf.float32)
        periodicity =  tf.cast(tf.argmax(periodicity, axis=-1),dtype=tf.float32)
        loss1 = mae( y_true1, periodicity)
        loss2 = bce(y_true2, with_in_period)
        loss = loss1 + loss2
        loss_tracker.update_state(loss)
        loss_tracker1.update_state(loss1)
        loss_tracker2.update_state(loss2)
        metric_tracker3.update_state(y_true1,periodicity)
        metric_tracker2.update_state(y_true2, with_in_period)
        return {'loss': loss_tracker.result(),
                'periodicity_loss': loss_tracker1.result(),
                'with_in_periody_loss': loss_tracker2.result(),
                'periodicity_acc': metric_tracker3.result(),
                'with_in_period_acc': metric_tracker2.result()}
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker,loss_tracker1,loss_tracker2,metric_tracker3,metric_tracker2]