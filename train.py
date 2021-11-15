from tensorflow import keras
from Datasets.utility import load_dataset
from model.PeriodEstimator import rep_net_mobile
from datetime import datetime
import os


import tensorflow as tf


tf.executing_eagerly()
SAMPLE_SIZE = 6000
TEST_SAMPLE_SIZE = 100
BATCH_SIZE = 2
EPOCHS = 100
CURRENT_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
CHKPOINT_PATH = "ckpt/repnet-mbl-{epoch:04d}.ckpt"
LOG_DIR = "logs/" + CURRENT_TIME
TRAIN_DATA_PATH = 'G:/RepNet/RepNetData7/'
TEST_DATA_PATH = 'G:/RepNet/RepNetData9/'
INPUT_SHAPE = [BATCH_SIZE, 64, 112, 112, 3]

checkpoint_dir = os.path.dirname(CHKPOINT_PATH)
train_dataset = load_dataset(TRAIN_DATA_PATH,sample_size=SAMPLE_SIZE,mode ='all', onehot=False).take(SAMPLE_SIZE).repeat(EPOCHS).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = load_dataset(TEST_DATA_PATH,sample_size=TEST_SAMPLE_SIZE,mode ='all', onehot=False).take(TEST_SAMPLE_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = rep_net_mobile(input_shape=INPUT_SHAPE)
model.summary()
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest:
    model.load_weights(latest)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=6e-6),
    run_eagerly=True
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current EPOCHS.
        filepath=CHKPOINT_PATH,
        save_weights_only=True,
    ),
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, update_freq=1),
]

model.fit(train_dataset,steps_per_epoch=SAMPLE_SIZE/BATCH_SIZE, epochs=EPOCHS , callbacks=callbacks, 
validation_data=test_dataset
)

