import imp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from .transformer import TransformerLayer
# from RepNet import TransformerLayer

def batched_pairwise_l2_distance(A, B):
    """
    (a-b)^2 = a^2 -2ab + b^2
    A shape = (N, D)
    B shaep = (C, D)
    result shape = (N, C)
    """
    batch = tf.shape(A)[0]
    row_norms_A = tf.math.reduce_sum(tf.square(A), axis=-1)
    row_norms_A = tf.reshape(row_norms_A, [batch, -1, 1])  # Column vector.

    row_norms_B = tf.math.reduce_sum(tf.square(B), axis=-1)
    row_norms_B = tf.reshape(row_norms_B, [batch, 1, -1])  # Row vector.
    dist = row_norms_A - 2. * tf.matmul(A, B, transpose_b=True) + row_norms_B
    return tf.math.maximum(dist, 0.)
#Convolutional feature extractor
def backbone(x, name='MobileNet'):
    model = tf.keras.applications.MobileNetV2(input_shape=x.shape[1:] ,include_top=False)
    model.trainable = False
    return model

def feature_encoder(x, trainalbe = True,backbone_shape=(96,96)):
    input = tf.keras.Input(x.shape[1:], batch_size=x.shape[0],name='feature_encoder_input')
    y = tf.reshape(input, [-1, input.shape[2], input.shape[3], input.shape[4]])
    y = tf.image.resize(y, backbone_shape)
    y = tf.keras.applications.mobilenet_v2.preprocess_input(y)
    y = backbone(y)(y)
    h = tf.shape(y)[1]
    w = tf.shape(y)[2]
    c = tf.shape(y)[3]
    y = tf.reshape(y, [x.shape[0], -1, h, w, c])
    feature_encoder = tf.keras.Model(input,y,name='feature_encoder')
    feature_encoder.trainable = trainalbe
    return feature_encoder

#Tenporal context and dimensionality reduction
def reduce_temporal_feature(x,
                     channels=512,
                     kernel_size=3,
                     dilation=3,
                     l2_reg_weight=1e-6,
                     trainable=True):
    """
    Conv3D -> GlobalMaxpooling2D
    """
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0],name='reduce_temporal_feature_input')
    y = tf.keras.layers.Conv3D(channels, kernel_size, padding='same',
                               dilation_rate=(dilation, 1, 1),
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg_weight),
                               kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.reduce_max(y, [2, 3])
    feature_reduction = tf.keras.Model(x1, y, name='reduce_temporal_feature')
    feature_reduction.trainable= trainable
    return feature_reduction

def self_similarity(x, temperature=13.544, trainable = True):
    """Calculates self-similarity between batch of sequence of embeddings."""
    x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0], name='self_similarity_input')
    y = -1.0 * batched_pairwise_l2_distance(x1, x1)
    y /= temperature
    y = tf.nn.softmax(y, axis=-1)
    y = tf.expand_dims(y, -1)
    similarity = tf.keras.Model(x1, y, name='self_similarity')
    similarity.trainable = trainable
    return similarity

def period_embedding(x,conv_channels=32, trainable=True):
    """
        Conv2D
    """
    # (batch_size, num_frames, num_frames, 1)
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0], name='period_embedding_input')
    y = tf.keras.layers.Conv2D(conv_channels, 3, padding='same',activation=tf.nn.relu)(y)
    # y = tf.keras.layers.ReLU()(y)
    y = tf.reshape(y, [x.shape[0], x.shape[1], x.shape[1] * conv_channels])

    period_embeder = tf.keras.Model(x1, y, name='period_embedding')
    period_embeder.trainable = trainable
    return period_embeder

def periodicity_classifier(x,                     
                           d_model=512, 
                           n_heads=4, 
                           dff=512, 
                           fc_channels=512, 
                           dropout=0.25,
                           trainable=True,
                           num_classes=32):
    """
        FC -> Transformer -> FC -> FC -> FC
    """
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0], name='periodicity_classifier_input')
    y = tf.keras.layers.Dense(d_model)(y) # (batch_size, num_frames, d_model)
    y = TransformerLayer(d_model, n_heads, dff, x.shape[1])(y)   
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dense(num_classes,activation=tf.nn.softmax)(y)
    periodicity = tf.keras.Model(x1, y, name='periodicity_output')
    periodicity.trainable =trainable
    return periodicity

def periodicity_regression(x,                     
                           d_model=512, 
                           n_heads=4, 
                           dff=512, 
                           fc_channels=512, 
                           dropout=0.25,
                           trainable=True):
    """
        FC -> Transformer -> FC -> FC -> FC
    """
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0], name='periodicity_classifier_input')
    y = tf.keras.layers.Dense(d_model)(y) # (batch_size, num_frames, d_model)
    y = TransformerLayer(d_model, n_heads, dff, x.shape[1])(y)   
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dense(1)(y)
    periodicity = tf.keras.Model(x1, y, name='periodicity_output')
    periodicity.trainable =trainable
    return periodicity

def with_in_period_classifier(x,                     
                           d_model=512, 
                           n_heads=4, 
                           dff=512, 
                           fc_channels=512, 
                           dropout=0.25,
                           trainable=True):
    """
        FC -> Transformer -> FC -> FC -> FC
    """
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0], name='with_in_period_classifier_input')
    y = tf.keras.layers.Dense(d_model)(y) # (batch_size, num_frames, d_model)
    y = TransformerLayer(d_model, n_heads, dff, x.shape[1])(y)  
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dense(1)(y)
    with_in_period = tf.keras.Model(x1, y, name='with_in_period_output')
    with_in_period.trainable = trainable
    return with_in_period

class FeatureEncoder(keras.layers.Layer):
    def __init__(self,input_shape=(96,96,3)):
        super(FeatureEncoder, self, ).__init__()
        backbone = tf.keras.applications.MobileNetV2(input_shape=input_shape,include_top=False)
        backbone.trainable = False
        self.backbone = tf.keras.models.Model(
            inputs=backbone.input,
            outputs=backbone.output)
    def call(self, x):
        y = tf.reshape(x, [-1, x.shape[2], x.shape[3], 3])
        y = tf.image.resize(y, (96, 96))
        y = self.backbone(y)
        h = tf.shape(y)[1]
        w = tf.shape(y)[2]
        c = tf.shape(y)[3]
        y = tf.reshape(y, [x.shape[0], -1, h, w, c])
        return y

class ReduceTemporalFeature(keras.layers.Layer):
    def __init__(self,channels=512,
                     kernel_size=3,
                     dilation=3,
                     l2_reg_weight=1e-6):
        super(ReduceTemporalFeature, self, ).__init__()
        self.channels=channels,
        self.kernel_size=kernel_size,
        self.dilation=dilation,
        self.l2_reg_weight=l2_reg_weight,
    def call(self, x):
        y = tf.keras.layers.Conv3D(self.channels, self.kernel_size, padding='same',
                                dilation_rate=(self.dilation, 1, 1),
                                kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                                kernel_initializer='he_normal')(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.reduce_max(y, [2, 3])
        return y

class SelfSimilarity(keras.layers.Layer):
    def __init__(self,temperature=13.544):
        super(SelfSimilarity, self).__init__()
        self.temperature = temperature
    def call(self, x):
        y = -1.0 * batched_pairwise_l2_distance(x, x)
        y /= self.temperature
        y = tf.nn.softmax(y, axis=-1)
        y = tf.expand_dims(y, -1)
        return y

class PeriodEmbedding(keras.layers.Layer):
    def __init__(self,conv_channels=32):
        super(PeriodEmbedding, self).__init__()
        self.conv_channels = conv_channels
    def call(self, x):
        y = tf.keras.layers.Conv2D(self.conv_channels, 3, padding='same',activation=tf.nn.relu)(x)
        y = tf.reshape(y, [x.shape[0], x.shape[1], x.shape[1] * self.conv_channels])
        return y

class PeriodicityClassifier(keras.layers.Layer):
    def __init__(self,  d_model=512, 
                        n_heads=4, 
                        dff=512, 
                        fc_channels=512, 
                        dropout=0.25,
                        trainable=True,
                        num_classes=32,
                        l2_regularizers=1e-6):
        super(PeriodicityClassifier, self).__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.dff=dff
        self.fc_channels=fc_channels
        self.dropout=dropout
        self.trainable=trainable
        self.num_classes=num_classes
        self.l2_regularizers = l2_regularizers
        

    def call(self, x):
        y = tf.keras.layers.Dense(self.d_model)(x) # (batch_size, num_frames, d_model)
        y = TransformerLayer(self.d_model, self.n_heads, self.dff, x.shape[1])(y)   
        y = tf.keras.layers.Dropout(self.dropout)(y)
        y = tf.keras.layers.Dense(self.fc_channels,
                kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Dropout(self.dropout)(y)
        y = tf.keras.layers.Dense(self.fc_channels,
                kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Dense(self.num_classes)(y)
        return y

class WithInPeriodClassifier(keras.layers.Layer):
    def __init__(self,  d_model=512, 
                        n_heads=4, 
                        dff=512, 
                        fc_channels=512, 
                        dropout=0.25,
                        trainable=True,
                        num_classes=1,
                        l2_regularizers=1e-6):
        super(WithInPeriodClassifier, self).__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.dff=dff
        self.fc_channels=fc_channels
        self.dropout=dropout
        self.trainable=trainable
        self.num_classes=num_classes
        self.l2_regularizers = l2_regularizers
    def call(self, x):
        y = tf.keras.layers.Dense(self.d_model)(x) # (batch_size, num_frames, d_model)
        y = TransformerLayer(self.d_model, self.n_heads, self.dff, x.shape[1])(y)   
        y = tf.keras.layers.Dropout(self.dropout)(y)
        y = tf.keras.layers.Dense(self.fc_channels,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularizers))(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Dropout(self.dropout)(y)
        y = tf.keras.layers.Dense(self.fc_channels,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularizers))(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Dense(self.num_classes)(y)
        return y






