from .opt_callbacks import optimizer_h5
from .layers_loss_metrics import GeMPoolingLayer,ArcFace,ArcFaceLoss_Metrics

import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.python.util import nest

from tensorflow.keras import applications,layers,Sequential,Model,regularizers as reg


models = {"DenseNet201":applications.DenseNet201,
          "Xception":applications.Xception,
          "ResNet152V2":applications.ResNet152V2,
          "EfficientNetB7":efn.EfficientNetB7,
          "EfficientNetB6":efn.EfficientNetB6,
          "EfficientNetB5":efn.EfficientNetB5}
poolings = {
    "mean":layers.GlobalAveragePooling2D(),
    "max":layers.GlobalMaxPool2D(),
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

        inp = layers.Input(shape=(self.height,self.width, 3))
        pretrained_model = models[name](weights = weights, include_top = False,
                            input_shape = [self.height,self.width, 3], input_tensor = inp)
        model = Sequential([
            inp, pretrained_model,self.poolings[pool],
            layers.Dense(self.embed_size, activation=None, kernel_initializer="glorot_normal",
                        dtype=tf.float32,name = "feature"+suffix),
            ArcFace(nclasses,dtype=tf.float32,name = "ArcFace"+suffix)
        ],name="%s_%s_ArcFace"%(name,pool))
        return model

    def transfer_model(self,nclasses1,nclasses2,path,name="EfficientNetB6",pool="gem",suffix = ""):
        inp = layers.Input(shape=(self.height,self.width, 3))
        pretrained_model = models[name](weights = None, include_top = False,
                         input_shape = [self.height,self.width, 3], input_tensor = inp)
        model = Sequential([
            inp, pretrained_model,self.poolings[pool],
            layers.Dense(self.embed_size, activation=None, kernel_initializer="glorot_normal",
                        dtype=tf.float32,name = "feature"+suffix),
            ArcFace(nclasses1,dtype=tf.float32,name = "ArcFace"+suffix)
        ])
        model.load_weights(path)

        model = Sequential([
            *model.layers[:-1],
            ArcFace(nclasses2,dtype=tf.float32,name = "ArcFace"+suffix)
        ],name = "%s_%s_ArcFace_more"%(name,pool))
        return model

def recursive_get_layer(model,name):
    try:
        layer = model.get_layer(name=name)
        return layer
    except ValueError:
        models = [m for m in model.layers if isinstance(m,Model)]
        for m in models:
            layer = recursive_get_layer(m,name)
            if layer is not None: return layer
        return None
    return None

@tf.function
def std_mean(ts,axis=None):
    shape = tf.cast(ts.shape,tf.float32)
    if axis is not None:
        shape = tf.gather(shape, axis)
    sqrt_n = tf.sqrt(tf.reduce_prod(shape))
    res = tf.multiply(tf.reduce_mean(ts,axis),sqrt_n)
    return res


class Branches_builder:
    @staticmethod
    def build_outclass_detector(embed,nclass=1):
        model = Sequential([
            layers.Input(shape=(embed,)),
            layers.Dense(nclass, activation="softmax", 
                        dtype=tf.float32,name = "inclass"),
        ],name="outclass_detector")
        return model

class DELG_attention:
    def __init__(self,encoder_filters=128, ae_reg=True, arcface = True, 
                    attention_filters=512,kernel_size=1,decay=0.0001):
            self.encoder_filters = encoder_filters
            self.ae_reg = ae_reg
            self.arcface = arcface
            self.attention_filters = attention_filters
            self.kernel_size = kernel_size
            self.decay = decay

    def build_model(self,shape,nclass,direct=False):
        decoder_filters = shape[-1]

        inp = layers.Input(shape=shape)
        conv1 = layers.Conv2D(
            self.attention_filters,self.kernel_size,
            kernel_regularizer=reg.l2(self.decay),
            padding='same', name='attn_conv1')(inp)
        bn_conv1 = layers.BatchNormalization(axis=3, name='bn_conv1')(conv1)

        conv2 = layers.Conv2D(
            1, self.kernel_size,
            kernel_regularizer=reg.l2(self.decay),
            padding='same', name='attn_conv2')(bn_conv1)
        attn_score = layers.Activation('softplus',name="attn_score",dtype=tf.float32)(conv2)

        encode = layers.Conv2D(
            self.encoder_filters, 1,
            kernel_regularizer = reg.l2(self.decay) if self.ae_reg else None,
            padding='same', name='auto_encoder',dtype=tf.float32)(inp)

        decode = layers.Conv2D(
            decoder_filters, 1,
            kernel_regularizer = reg.l2(self.decay) if self.ae_reg else None,
            padding='same', name='auto_decoder')(encode)
        decode_activation = layers.Activation('swish',name="decoder_out")(decode)

        mean = layers.Lambda(
            lambda x: std_mean(x,[1,2,3])
            ,name="mean_decoder_out") (decode_activation)

        norm = layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=-1)
            ,name="descriptor")

        if direct:
            norm = norm(encode)
        else:
            norm = norm(decode_activation)

        feat = layers.Lambda(
            lambda ls: tf.reduce_mean(tf.multiply(ls[0], ls[1]), [1, 2], keepdims=False)
            ,name="mean_descriptor",dtype=tf.float32)([norm,attn_score])

        if self.arcface:
            feat = ArcFace(nclass,dtype=tf.float32,name = "result")(feat)
        else:
            feat = layers.Dense(nclass,dtype=tf.float32,activation="softmax", name='result')(feat)

        model = Model(inputs=inp, outputs = [mean,feat],name="DELG_attn")
        return model,norm.output_shape

    def build_sep_training(self,stem,shape,nclass,direct=False,train_weight=False,
            valid_weight=True,input_layer_name="block6a_expand_activation",
            clip_norm = None):
        branch = self.build_model(shape,nclass,direct)
        self.model,output_shape = Model_w_AE_on_single_middle_layer(stem,branch,
                input_layer_name=input_layer_name,
                out_type=["auto_encoder","normal"],
                train_weight=train_weight,
                valid_weight=valid_weight,
                clip_norm = clip_norm)
        return self.model,output_shape

    def export_branch(self):
        try:
            model = self.__getattribute__("model")
        except:
            raise ValueError("No model defined.")

        return self.export_model(model.branch)

    @staticmethod
    def export_model(model,names=None):
        if names is None: 
            names = ["descriptor","attn_score"]
        elif not isinstance(names,(list,tuple)):
            names = [names]

        model = Model(inputs = model.input, outputs = 
            [recursive_get_layer(model,n) for n in names]
        ,name="DELG_attn_export")
        return model

class Model_w_AE_on_single_middle_layer(Model):
    dic_type_2num = {
        "normal": 1,
        "auto_encoder": 0,
    }
    def __init__(self, stem, branch, input_layer_name,out_type,
                 subsample=True,train_weight=False, valid_weight=False,
                 clip_norm = None):
        super(Model_w_AE_on_single_middle_layer, self).__init__()
        self.branch = branch
        self.subsample = subsample
        self.clip_norm = clip_norm
        self._transfer_stem_model(stem,input_layer_name)
        self.num_outs = len(branch.outputs)

        if not isinstance(out_type,(list,tuple)): out_type=[out_type]

        assert self.num_outs==len(out_type)
        if any(i not in self.dic_type_2num for i in out_type):
            raise ValueError("out_type must be one of {}".format(self.dic_type_2num.keys()))

        self.out_type = out_type
        self.train_weight = train_weight
        self.valid_weight = valid_weight   

    def _transfer_stem_model(self,stem,input_layer_name):
        try:
            layer = recursive_get_layer(stem,input_layer_name).get_output_at(-1)
            if self.subsample: 
              layer = layers.MaxPooling2D( (1, 1), strides=(2, 2))(layer)
            self.stem = Model(inputs=stem.inputs, outputs = layer,name = "middle_"+stem.name)
            self.stem.trainable = False
        except:
            print("The model with nested sub-model inside suffers from multiple"
                " bound nodes problems. Please revise the nest model and change stem"
                " model input & output to the outtest one. (tf.keras.layers.Input layer"
                " ahead may help) Details check https://github.com/tensorflow/tensorflow/issues/34977.")
            raise  

    def sep_data(self,data,w):
        main_input = data[0]
        if w: weight,data = tf.reshape(data[-1],[-1,1]),data[:-1]
        else: weight = None
        assert len(data) == 1+sum(self.dic_type_2num[i] for i in self.out_type)

        labels,weights = [],[weight]*self.num_outs
        label_id = 0
        for key in self.out_type:
            label = None
            if key == "normal":
                label_id += 1
                label = data[label_id]
            labels.append(label)

        return main_input,labels,weights

    def train_step(self, data):
        main_input, labels,weights = self.sep_data(data,self.train_weight)

        with tf.GradientTape() as tape:
            middle,res = self(main_input,training=True)
            labels = [middle if i is None else i for i in labels]
            loss = self.compiled_loss(labels, res, sample_weight=weights)

        variable = self.branch.trainable_weights
        grads = tape.gradient(loss, variable)
        if self.clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clip_norm)

        self.optimizer.apply_gradients(zip(grads, variable))
        self.compiled_metrics.update_state(labels, res, sample_weight=weights)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        main_input, labels,weights = self.sep_data(data,self.valid_weight)
        middle,res = self(main_input,training=False)
        labels = [middle if i is None else i for i in labels]

        self.compiled_loss(labels, res, sample_weight=weights)
        self.compiled_metrics.update_state(labels, res, sample_weight=weights)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None):
        middle = self.stem(inputs,training=False)
        res = self.branch(middle,training=training)
        return std_mean(middle,[1,2,3]),res


class Transfer_builder:
    @staticmethod
    def flatten_output(outs):
        res = []
        for i in outs:
            if isinstance(i,(list,tuple)):
                res.append(Transfer_builder.flatten_output(i))
            else:
                res.append(i)
        return res

    @staticmethod
    def transfer_stem_model(stem,branch,name):
        in_layer = recursive_get_layer(stem,name).get_output_at(-1) 
        out = branch(in_layer)
        return Model(inputs=stem.inputs, outputs = out,
                            name="%s_%s"%(stem.name,branch.name))
    @staticmethod
    def transfer_multiple_model(stem,branches,names,include_stem=True):

        in_layers = [recursive_get_layer(stem,name).get_output_at(-1) 
                for name in names]
        outs = [branch(l) for l,branch in zip(in_layers,branches)]
        outs = Transfer_builder.flatten_output(outs)
        stem_out = stem.outputs if include_stem else []
        return Model(inputs=stem.inputs, outputs = stem_out + outs,
                            name="%s_%s"%(stem.name,"Multiple_output"))

if tf.__version__>="2.2.0":
    from tensorflow.python.keras.engine import compile_utils
    class Model_w_self_backpropagated_branches(Model):
        dic_type_2num = {
            "normal": 1,
            "normal_weight": 2,
            "auto_encoder": 0,
            "auto_encoder_weight": 1
        }

        def __init__(self, stem, branches, input_layer_names,
                    train_type, valid_type):
            super(Model_w_self_backpropagated_branches, self).__init__()
            self.branches = branches
            self.input_layer_names = input_layer_names

            assert len(branches)==len(input_layer_names)
            assert len(branches)+1==len(train_type)
            assert len(branches)+1==len(valid_type)

            self.num_models = len(self.branches) + 1
            self._transfer_stem_model(stem)

            if (any(i not in self.dic_type_2num for i in train_type) or 
                any(i not in self.dic_type_2num for i in valid_type)):
                raise ValueError("train_type and valid_type must be one of {}".
                                format(self.dic_type_2num.keys()))

            self.train_type = train_type
            self.valid_type = valid_type

        def _transfer_stem_model(self,stem):
            try:
                name_layer_unique= list(set(self.input_layer_names))
                self.input_index = [name_layer_unique.index(i) for i in self.input_layer_names]


                layers = [ recursive_get_layer(stem,n).get_output_at(-1) for n in name_layer_unique]
                self.stem = Model(inputs=stem.inputs, outputs = stem.outputs + layers,
                    name = stem.name + "_multi_output")
            except:
                print("The model with nested sub-model inside suffers from multiple"
                    " bound nodes problems. Please revise the nest model and change stem"
                    " model input & output to the outtest one. (tf.keras.layers.Input layer"
                    " ahead may help) Details check https://github.com/tensorflow/tensorflow/issues/34977.")
                raise

        def sep_input(self,data,types):
            assert len(data) == 1 + sum(self.dic_type_2num[i] for i in types)
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
                self._validate_compile(optimizers, user_metrics)

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
                metrics += nest.flatten([l.metrics for l in self.compiled_loss])
            if self.compiled_metrics is not None:
                metrics += nest.flatten([m.metrics for m in self.compiled_metrics])

            for l in self._flatten_layers():
                metrics.extend(l._metrics)  # pylint: disable=protected-access
            return metrics


        def train_step(self, data):
            label_weights = self.sep_input(data,self.train_type)

            inputs = []
            for i,lw in enumerate(label_weights):
                ts_input = data[0] if i==0 else inputs[1+self.input_index[i-1]]
                model = self.stem if i==0 else self.branches[i-1]
                label = lw.get("label", ts_input)
                sample_weight = lw.get("weight",None)

            
                with tf.GradientTape() as tape:
                    if i==0:
                        inputs = model(ts_input,training=True)
                        predictions = inputs[0]
                    else:
                        predictions = model(ts_input,training=True)
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
            label_weights = self.sep_input(data,self.valid_type)
            inputs = self.stem(data[0],training=False)

            for i,lw in enumerate(label_weights):
                if i==0:
                    ts_input,y_pred = data[0],inputs[0]
                else:
                    ts_input = inputs[1+self.input_index[i-1]]
                    y_pred = self.branches[i-1](ts_input,training=False)

                label = lw.get("label", ts_input)
                sample_weight = lw.get("weight",None)

                self.compiled_loss[i](label, y_pred, sample_weight=sample_weight)
                self.compiled_metrics[i].update_state(label, y_pred, sample_weight=sample_weight)

            return {m.name: m.result() for m in self.metrics}

        def call(self, inputs, training=None, mask=None):
            inputs = self.stem(inputs,training=training)
            res,inputs = inputs[:1],inputs[1:]

            for i in range(self.num_models-1):
                ts_input = inputs[self.input_index[i]]
                res.append(self.branches[i](ts_input,training=training))

            return res