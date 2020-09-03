import tensorflow as tf
import math

class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., train_p=False,eps = 1e-6):
        super().__init__()
        self.p = p
        self.train_p=train_p
        self.eps = eps

    def build(self, input_shape):
        if self.train_p:
            self.pow = tf.Variable(self.p, dtype=tf.float32)
        else:
            self.pow = self.p
        super(GeMPoolingLayer, self).build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(inputs, clip_value_min=self.eps, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.pow)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1./self.pow)
        return inputs
    
    def get_config(self):
        return {"p":self.p,"train_p":self.train_p,"eps":self.eps}

class ArcFace(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
    def get_config(self):
        return {"num_classes":self.num_classes}

    def build(self, input_shape):
        self.w = self.add_weight("weights", shape=[input_shape[-1], self.num_classes],
                                 initializer='glorot_normal',trainable=True)
        super(ArcFace, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

    def call(self, x):
        embeddings = tf.nn.l2_normalize(x, axis=1, name='normed_embeddings')
        weights = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')
        cos_t = tf.matmul(embeddings, weights, name='cos_t')
        return cos_t

class ArcFaceLoss_Metrics():
    def __init__(self, num_classes, margin=0.5, logit_scale=64.,metrics=False):
        self.num_classes = num_classes
        self.margin = margin
        self.logit_scale = logit_scale
        self.metrics = metrics
        self.__name__ = "ArcFace" + ("Accuracy" if metrics else "Loss")
        
        self.cos_m = tf.identity(tf.cos(margin), name='cos_m')
        self.sin_m = tf.identity(tf.sin(margin), name='sin_m')
        self.th = tf.identity(tf.cos(math.pi - margin), name='th')
        self.mm = tf.multiply(self.sin_m, margin, name='mm')
        
    def get_config(self):
        return {"num_classes":self.num_classes,"margin":self.margin,
                "logit_scale":self.logit_scale,"metrics":self.metrics}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_logits(self, labels, cos_t):
        sin_t = tf.sqrt(tf.subtract(1., tf.square(cos_t)), name='sin_t')
        cos_mt = tf.subtract(tf.multiply(cos_t, self.cos_m), tf.multiply(sin_t, self.sin_m), name='cos_mt')
        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)
        mask = tf.one_hot(labels , depth=self.num_classes, name='one_hot_mask',
                         on_value=True, off_value=False, dtype=tf.bool)
        logits = tf.where(mask, cos_mt, cos_t)
        logits = tf.multiply(logits, self.logit_scale, 'arcface_logist')
        return logits

    def __call__(self, y_true, y_pred):
        labels = tf.cast(tf.reshape(y_true,[-1]),tf.int32)
        logits = self.get_logits(labels, y_pred)
        if self.metrics:
            res = tf.keras.metrics.sparse_categorical_accuracy(labels,logits)
        else:
            res = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        return res
