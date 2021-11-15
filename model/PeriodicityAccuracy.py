import tensorflow as tf

class PeriodicityAccuracy(tf.keras.metrics.Metric):

  def __init__(self,num_classes,name='periodicity_acccuracy', **kwargs):
    super(PeriodicityAccuracy, self).__init__(name=name, **kwargs)
    self.num_classes = num_classes
    self.total_cm = self.add_weight(name='total_cm', initializer='zeros', shape=(num_classes,num_classes))
    # self.periodicity_acccuracy = self.add_weight(name='pa', initializer='zeros')

  def update_state(self, y_true, y_pred):
    self.total_cm.assign_add(self.comfusion_matrix(y_true, y_pred))
    return self.total_cm

  def reset_state(self):
    for s in self.variables:
        s.assign(tf.zeros(shape=s.shape))

  def comfusion_matrix(self, y_true, y_pred):
    idxs = tf.cast(tf.reduce_max(y_true, axis=-1),dtype=tf.int64)
    idxs = tf.reshape(idxs,shape=[idxs.shape[0]*idxs.shape[1]])
    idxs_1 =tf.where(tf.math.greater(idxs,0))

    labels_max = tf.squeeze(tf.gather(tf.reshape(tf.argmax(y_true, axis=-1),shape=[idxs.shape[0]]), idxs_1),-1)
    logits_max =tf.squeeze(tf.gather(tf.reshape(tf.argmax(y_pred, axis=-1),shape=[idxs.shape[0]]), idxs_1),-1)
    cm =  tf.cast(tf.math.confusion_matrix(labels_max, logits_max, num_classes=self.num_classes), tf.float32)
    return cm
  
  def process_confusion_matrix(self):
    cm = self.total_cm
    diag_part = tf.cast(tf.linalg.diag_part(cm),tf.float32)
    precision = diag_part/(tf.cast(tf.reduce_sum(cm,0),tf.float32)+1e-15)
    recall = diag_part/(tf.cast(tf.reduce_sum(cm,1),tf.float32)+1e-15)
    f1= 2*precision*recall/(precision+recall+1e-15)
    f1 = tf.reduce_mean(f1)
    return f1

  def result(self):
    return self.process_confusion_matrix()

  def fill_output(self, output):
    result = self.result()
    for i in range(self.num_classes):
        output['precision:{}'.format(i)] = result[0][i]
        output['recall:{}'.format(i)] = result[1][i]
        output['F1:{}'.format(i)] = result[2][i]