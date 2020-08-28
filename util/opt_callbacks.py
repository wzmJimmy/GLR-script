from tensorflow.python.keras.saving.hdf5_format import load_optimizer_weights_from_hdf5_group,save_optimizer_weights_to_hdf5_group
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.platform import tf_logging as logging

import six

try:
  import h5py
  HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
  h5py = None

class optimizer_h5():
    @staticmethod
    def load_optimizer_from_hdf5(filepath, model):
        with h5py.File(filepath, mode='r') as f:
            if 'optimizer_weights' in f:
                try:
                    model.optimizer._create_all_weights(model.trainable_variables)
                except (NotImplementedError, AttributeError):
                    logging.warning('Error when creating the weights of optimizer {}')

                optimizer_weight_values = load_optimizer_weights_from_hdf5_group(f)
                try:
                    model.optimizer.set_weights(optimizer_weight_values)
                except ValueError:
                    logging.warning('Error in loading the saved optimizer state.')
                    
    @staticmethod
    def save_optimizer_to_hdf5(model, filepath):
        with h5py.File(filepath, mode='w') as f:
            if (model.optimizer and not isinstance(model.optimizer, optimizers.TFOptimizer)):
                save_optimizer_weights_to_hdf5_group(f, model.optimizer)
            f.flush()

from tensorflow.python.keras.callbacks import ProgbarLogger,ModelCheckpoint,CallbackList
from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import Progbar

class NBatchProgBarLogger(ProgbarLogger):
    def __init__(self, count_mode='steps', stateful_metrics=None, display_per_batches=100):
        super(NBatchProgBarLogger, self).__init__(count_mode, stateful_metrics)
        self.display_per_batches = display_per_batches
        self.display_step = 1
        
    def _batch_update_progbar(self, batch, logs=None):
        """Updates the progbar."""
        logs = logs or {}
        self._maybe_init_progbar()

        if self.use_steps:
            self.seen = batch + 1  # One-indexed.
        else:
            batch_size = logs.get('size', 0)
            num_steps = logs.get('num_steps', 1)
            self.seen += batch_size * num_steps
        
        self.display_step += 1
        if (self.verbose == 1 and self.seen < self.target and 
            self.display_step % self.display_per_batches == 0):
            logs = tf_utils.to_numpy_or_python_type(logs)
            self.progbar.update(self.seen, list(logs.items()), finalize=False)

    def _reset_progbar(self):
        self.seen = 0
        self.progbar = None
        self.display_step = 1

class NEpochModelCheckpoint_wOptimizer(ModelCheckpoint):
    def __init__(self,filepath,nepoch=3,opt=False,**kwargs):
        super(NEpochModelCheckpoint_wOptimizer, self).__init__(filepath,**kwargs)
        self.opt = opt
        self.nepoch = nepoch
        self.save_format = 'h5' if ("h5" in filepath or "hdf5" in filepath) else "tf"

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save==self.nepoch:
            self.epochs_since_last_save = 0
            self._save_model(epoch=epoch, logs=logs)

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
            file_path = self.filepath.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                        'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        if self.opt:
            li = file_path.split(".")
            li[-2] += "_opt"
            filr_path_opt = ".".join(li)
            self._write_filepath_opt = distributed_file_utils.write_filepath(
                filr_path_opt, self.model.distribute_strategy)
        return self._write_filepath

    def _maybe_remove_file(self):
        distributed_file_utils.remove_temp_dir_with_filepath(
            self._write_filepath, self.model.distribute_strategy)
        if self.opt: 
            distributed_file_utils.remove_temp_dir_with_filepath(
                self._write_filepath_opt, self.model.distribute_strategy)
    
    def _save_model(self, epoch, logs):
        logs = logs or {}
        logs = tf_utils.to_numpy_or_python_type(logs)
        filepath = self._get_file_path(epoch, logs)
        
        try:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor,
                                                          self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True, options=self._options)
                            if self.opt: optimizer_h5.save_optimizer_to_hdf5(self.model, self._write_filepath_opt) ##
                        else:
                            self.model.save(filepath, overwrite=True, options=self._options,
                                            save_format = self.save_format)
                    elif self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True, options=self._options)
                    if self.opt: optimizer_h5.save_optimizer_to_hdf5(self.model, self._write_filepath_opt) ##
                else:
                    self.model.save(filepath, overwrite=True, options=self._options,
                                    save_format = self.save_format)

            self._maybe_remove_file()
        except IOError as e:
        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
            if 'is a directory' in six.ensure_str(e.args[0]).lower():
                raise IOError('Please specify a non-directory filepath for '
                                'ModelCheckpoint. Filepath used is an existing '
                                'directory: {}'.format(filepath))

if __name__=="__main__":
    import numpy as np
    from tensorflow import keras
    from tensorflow.python.keras import layers

    num_classes = 10
    input_shape = (28, 28, 1)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    batch_size = 128
    epochs = 6
    steps_per_epoch = 5400//batch_size
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model_path = "exp-{epoch:02d}-{val_accuracy:.3f}.h5"
    mc = NEpochModelCheckpoint_wOptimizer(model_path, nepoch=3,opt=True,
                 monitor='val_accuracy', save_weights_only=True, mode='max',verbose=1)
    nbar = NBatchProgBarLogger(display_per_batches=10)
    callback_list = CallbackList([mc,nbar],add_history=True,add_progbar=False,
                model=model, verbose=1, epochs=epochs, steps=steps_per_epoch)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
            steps_per_epoch=steps_per_epoch,callbacks = callback_list)

    # model2 = keras.Sequential(
    #     [
    #         keras.Input(shape=input_shape),
    #         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #                 layers.Flatten(),
    #         layers.Dropout(0.5),
    #         layers.Dense(num_classes, activation="softmax"),
    #     ]
    # )
    # model2.load_weights("exp-06-0.969.h5")
    # model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # optimizer_h5.load_optimizer_from_hdf5("exp-06-0.969_opt.h5",model2)
    # print([i.name for i in model2.weights])
    # print([i.name for i in model2.optimizer.weights])