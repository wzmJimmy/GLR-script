from .opt_callbacks import optimizer_h5
from .layers_loss_metrics import GeMPoolingLayer,ArcFace,ArcFaceLoss_Metrics

import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.util import nest


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
        ],name = name+pool+"ArcFace")
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

class Branches_builder:
    @staticmethod
    def build_outclass_detector(embed,nclass=2):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(embed,)),
            tf.keras.layers.Dense(nclass, activation="softmax", 
                        dtype=tf.float32,name = "inclass"),
        ],name="outclass_detector")
        return model


class Model_w_self_backpropagated_branches(tf.keras.Model):
    dic_type_2num = {
        "normal": 1,
        "normal_weight": 2,
        "auto_encoder": 0,
        "auto_encoder_weight": 1
    }

    def __init__(self, stem, branches, input_layer_names,
                train_type, valid_type):
        super(Model_w_self_backpropagated_branches, self).__init__()
        self.stem = stem
        self.branches = branches
        self.input_layer_names = input_layer_names

        assert len(branches)==len(input_layer_names)
        assert len(branches)+1==len(train_type)
        assert len(branches)+1==len(valid_type)
        self.num_models = len(self.branches) + 1

        if (any(i not in self.dic_type_2num for i in train_type) or 
            any(i not in self.dic_type_2num for i in valid_type)):
            raise ValueError("train_type and valid_type must be one of {}".
                            format(self.get_type()))

        self.train_type = train_type
        self.valid_type = valid_type

    @staticmethod
    def get_type():
        return list(dic_type_2num.keys())

    @staticmethod
    def get_layer(model,name):
        try:
            layer = model.get_layer(name=name)
            return layer
        except ValueError:
            models = [m for m in model.layers if isinstance(m,tf.keras.Model)]
            for m in models:
                layer = Model_w_self_backpropagated_branches.get_layer(m,name)
                if layer is not None: return layer
            return None
        return None

    def _get_branches_input(self,idx):
        idx -= 1
        return self.get_layer(self.stem,name=self.input_layer_names[idx]).output

    def sep_input(self,data,types):
        assert self.num_models == 1 + sum(self.dic_type_2num(i) for i in types)
        inputs = [None]*self.num_models
        main_input = data[0]

        inputs,label_idx,weight_id = [],0,sum(key.startswith("normal") for key in types)
        for i,key in enumerate(types):
            dic = {}
            if key == "normal":
                label_idx += 1
                dic["label"] = data[label_idx]
            elif key == "normal_weight":
                label_idx += 1
                weight_id += 1
                dic["label"] = data[label_idx]
                dic["weight"] = data[weight_id]
            elif key == "auto_encoder_weight":
                weight_id += 1
                dic["weight"] = data[weight_id]

            if i==1: dic["input"] = main_input
            inputs.append(dic)

        return inputs

    def compile(self, optimizers, loss_fns, metrics=None):
        super(Model_w_self_backpropagated_branches, self).compile()
        assert len(optimizers)==len(loss_fns)
        assert len(optimizers)==self.num_models

        if metrics is None:
            user_metrics = [None]*self.num_models
        elif isinstance(metrics,dict):
            user_metrics = [None]*self.num_models
            for k,v in metrics.items():
                user_metrics[k] = v
        elif isinstance(metrics,list):
            assert len(metrics)==self.num_models
            user_metrics = metrics
        else:
            raise ValueError("Metrics must be either a full-list or"
                " a sparse version dictionary map position to metric.")

        with self.distribute_strategy.scope():
            self._validate_compile(optimizers, self.metrics)

            self.optimizers = self._get_optimizer(optimizers)
            self.compiled_loss = [compile_utils.LossesContainer(
                loss, output_names=self.output_names) 
                for loss in loss_fns]
            self.compiled_metrics = [compile_utils.MetricsContainer(
                metric, output_names=self.output_names)
                for metric in user_metrics]

    @property
    def metrics(self):
        metrics = []
        if self.compiled_loss is not None:
            metrics += nest.flatten(self.compiled_loss.metrics)
        if self.compiled_metrics is not None:
            metrics += nest.flatten(self.compiled_metrics.metrics)

        for l in self._flatten_layers():
            metrics.extend(l._metrics)  # pylint: disable=protected-access
        return metrics


    def train_step(self, data):
        inputs = self.sep_input(data,self.train_type)
        
        for i in range(self.num_models):
            ds_input = inputs[i]
            ts_input = ds_input.get("input", self._get_branches_input(i)) 
            label = ds_input.get("label", ts_input)
            sample_weight = ds_input.get("weight",None)
            model = self.stem if i==0 else self.branches[i-1]
            
            with tf.GradientTape() as tape:
                predictions = model(ts_input)
                loss = self.compiled_loss[i](
                    label, predictions,
                    sample_weight=sample_weight,
                )
            variable = model.trainable_weights
            grads = tape.gradient(loss, variable)
            self.optimizers[i].apply_gradients(zip(grads, variable))
            self.compiled_metrics[i].update_state(label, predictions, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        ts_input = data[0]
        y_preds = self(ts_input, training=False)

        inputs = self.sep_input(data,self.valid_type)
        for i in range(self.num_models):
            ds_input = inputs[i]
            label = ds_input.get("label", self._get_branches_input(i))
            sample_weight = ds_input.get("weight",None)
            self.compiled_loss[i](label, y_preds[i],
                    sample_weight=sample_weight)
            self.compiled_metrics[i].update_state(label, y_preds[i],
                    sample_weight=sample_weight)
        
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None, mask=None):
        res = []

        for i in range(self.num_models):
            if i==0:
                ts_input = inputs
                model = self.stem
            else:
                ts_input = self._get_branches_input(i)
                model = self.branches[i-1]
            res.append(model(ts_input,training=training))

        return res