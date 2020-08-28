from .opt_callbacks import optimizer_h5
from .layers_loss_metrics import GeMPoolingLayer,ArcFace,ArcFaceLoss_Metrics

import tensorflow as tf
import efficientnet.tfkeras as efn


models = {"DenseNet201":tf.keras.applications.DenseNet201,
          "Xception":tf.keras.applications.Xception,
          "ResNet152V2":tf.keras.applications.ResNet152V2,
          "EfficientNetB7":efn.EfficientNetB7,
          "EfficientNetB6":efn.EfficientNetB6,
          "EfficientNetB5":efn.EfficientNetB5}
poolings = {
    "mean":tf.keras.layers.GlobalAveragePooling2D(),
    "max":tf.keras.layers.GlobalMaxPool2D(),
    # "gem":GeMPoolingLayer(gem_p,train_p=train_p)
}

class Efn_Gem_Arc_builder:
    def __init__(self,height,width,embed_size=1024,gem_config=None):
        self.height = height
        self.width = width
        self.embed_size = embed_size
        self._config_poolings(gem_config or {})

    def _config_poolings(self,gem_config):
        gem_p = gem_config.get("gem_p",3.)
        train_p = gem_config.get("train_p",False)  
        self.poolings = poolings.copy()
        self.poolings["gem"] = GeMPoolingLayer(gem_p,train_p=train_p)

    def build_model(self,nclasses,name="EfficientNetB6",pool="gem",load=True,suffix = ""):
        weights = None
        if not load: 
            weights = 'noisy-student' if "EfficientNet" in name else 'imagenet'

        pretrained_model = models[name](weights = weights, include_top = False, input_shape = [self.height,self.width, 3])
        model = tf.keras.Sequential([
            pretrained_model,
            self.poolings[pool],
            tf.keras.layers.Dense(self.embed_size, activation=None, kernel_initializer="glorot_normal",
                        dtype=tf.float32,name = "feature"+suffix),
            ArcFace(nclasses,dtype=tf.float32,name = "ArcFace"+suffix)
        ])
        return model

    def transfer_model(self,nclasses1,nclasses2,path,name="EfficientNetB6",pool="gem",suffix = ""):
        pretrained_model = models[name](weights = None, include_top = False, input_shape = [self.height,self.width, 3])
        model = tf.keras.Sequential([
            pretrained_model,
            self.poolings[pool],
            tf.keras.layers.Dense(self.embed_size, activation=None, kernel_initializer="glorot_normal",
                        dtype=tf.float32,name = "feature"+suffix),
            ArcFace(nclasses1,dtype=tf.float32,name = "ArcFace"+suffix)
        ])
        model.load_weights(path)

        model = tf.keras.Sequential([
            *model.layers[:-1],
            ArcFace(nclasses2,dtype=tf.float32,name = "ArcFace"+suffix)
        ])
        return model