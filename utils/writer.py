import numpy as np
import tensorflow as tf

class MyWriter():
    def __init__(self, logdir):
        self.train_summary_writer = tf.summary.create_file_writer(logdir)

    def log_training(self,loss, periodicity_loss,with_in_periody_loss,periodicity_acc,with_in_period_acc,step):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('tarin_total_loss', loss, step)
            tf.summary.scalar('tarin_periodicity_loss', periodicity_loss, step)
            tf.summary.scalar('tarin_with_in_periody_loss', with_in_periody_loss, step)
            tf.summary.scalar('tarin_periodicity_acc', periodicity_acc, step)
            tf.summary.scalar('tarin_with_in_period_accs', with_in_period_acc, step)

    def log_evaluation(self,sim,loss, periodicity_loss,with_in_periody_loss,periodicity_acc,with_in_period_acc,step):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('test_total_loss', loss, step)
            tf.summary.scalar('test_periodicity_loss', periodicity_loss, step)
            tf.summary.scalar('test_with_in_periody_loss', with_in_periody_loss, step)
            tf.summary.scalar('test_periodicity_acc', periodicity_acc, step)
            tf.summary.scalar('test_with_in_period_accs', with_in_period_acc, step)
            tf.summary.image('similarity_matrix',sim, step)
