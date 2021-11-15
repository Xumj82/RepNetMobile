from tensorflow import keras
from Datasets.utility import load_dataset
from model.PeriodEstimator import rep_net_mobile
from datetime import datetime
import os

from model.PeriodicityAccuracy import PeriodicityAccuracy
import tensorflow as tf
from utils.writer import MyWriter

SAMPLE_SIZE = 6000
TEST_SAMPLE_SIZE = 100
BATCH_SIZE = 2
EPOCHS = 100
CURRENT_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
CHKPOINT_PATH = "ckpt/repnet-mbl.ckpt"
LOG_DIR = "logs/" + CURRENT_TIME
TRAIN_DATA_PATH = 'G:/RepNet/RepNetData7/'
TEST_DATA_PATH = 'G:/RepNet/RepNetData9/'
INPUT_SHAPE = [BATCH_SIZE, 64, 112, 112, 3]

checkpoint_dir = os.path.dirname(CHKPOINT_PATH)
train_dataset = load_dataset(TRAIN_DATA_PATH,sample_size=SAMPLE_SIZE,mode ='all', onehot=False).take(SAMPLE_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = load_dataset(TEST_DATA_PATH,sample_size=TEST_SAMPLE_SIZE,mode ='all', onehot=False).take(TEST_SAMPLE_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
log_writer = MyWriter(LOG_DIR)

model = rep_net_mobile(input_shape=INPUT_SHAPE)
model.summary()


if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt = tf.train.Checkpoint(model=model)
if tf.train.latest_checkpoint(checkpoint_dir):
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))

loss_tracker = tf.keras.metrics.Mean(name='loss')
loss_tracker1 = tf.keras.metrics.Mean(name='periodicity_loss')
loss_tracker2 = tf.keras.metrics.Mean(name='with_in_periody_loss')
metric_tracker1 = PeriodicityAccuracy(name='periodicity_acc', num_classes=32)
metric_tracker2 = tf.keras.metrics.BinaryAccuracy(name='with_in_period_acc')
metric_tracker3 = tf.keras.metrics.CosineSimilarity(name='periodicity_acc')
cce = tf.keras.losses.CategoricalCrossentropy()
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mae = tf.keras.losses.MeanAbsoluteError()
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=6e-6),
    # run_eagerly=True
)
# Iterate over epochs.
for epoch in range(EPOCHS):
    print("Start of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step,(x, y_true1, y_true2) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            periodicity, with_in_period, sim = model(x)
            loss1 = cce(y_true1, periodicity)
            loss2 = bce(y_true2, with_in_period)
            loss = loss1+loss2
        trainable_vars = model.trainable_weights
        grads = tape.gradient(loss, trainable_vars)
        model.optimizer.apply_gradients(zip(grads, trainable_vars))
        

        loss_tracker.update_state(loss)
        loss_tracker1.update_state(loss1)
        loss_tracker2.update_state(loss2)
        metric_tracker1.update_state(y_true1, periodicity)
        metric_tracker2.update_state(y_true2, with_in_period)
        log_writer.log_training(loss_tracker.result(),loss_tracker1.result(),loss_tracker2.result(),metric_tracker1.result(),metric_tracker2.result(),int(epoch*SAMPLE_SIZE/BATCH_SIZE+step))
        print("epoch {0} train step {1} --- periodicity_loss : {2}  with_in_periody_loss : {3}".format(epoch,step,loss_tracker1.result(),loss_tracker2.result()))
    for step,(x, y_true1, y_true2) in enumerate(test_dataset):
        periodicity, with_in_period,sim = model(x)
        loss1 = cce(y_true1, periodicity)
        loss2 = bce(y_true2, with_in_period)
        loss = loss1+loss2
        
        loss_tracker.update_state(loss)
        loss_tracker1.update_state(loss1)
        loss_tracker2.update_state(loss2)
        metric_tracker1.update_state(y_true1, periodicity)
        metric_tracker2.update_state(y_true2, with_in_period)
        log_writer.log_evaluation(sim, loss_tracker.result(),loss_tracker1.result(),loss_tracker2.result(),metric_tracker1.result(),metric_tracker2.result(),int(epoch*TEST_SAMPLE_SIZE/BATCH_SIZE+step))
        print("epoch {0} test step {1} --- periodicity_loss : {2}  with_in_periody_loss : {3}".format(epoch,step,loss_tracker1.result(),loss_tracker2.result()))

    ckpt.save('ckpt/repnet-mobile.ckpt',)
    print("Saved checkpoint for epoch {}".format(epoch))