import tensorflow as tf
import random
random.seed(1214)

FORMAT = {
        'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'label2': tf.io.FixedLenFeature([], tf.int64, default_value=1),
}
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False
AUTO = tf.data.experimental.AUTOTUNE

class Lookup:
    def __init__(self,clean_table = None,weight_table=None,
        weight_table_full=None,train_weight_table=None):
        self.clean_table = clean_table
        self.weight_table = weight_table
        self.weight_table_full = weight_table_full
        self.train_weight_table = train_weight_table

    @staticmethod
    def build_tf_hash_table(df,key,val,default,ktype='int32',vtype='int32'):
        key = tf.constant(df[key],dtype=ktype)
        val = tf.constant(df[val],dtype=vtype)
        init = tf.lookup.KeyValueTensorInitializer(key, val)
        table = tf.lookup.StaticHashTable(init, default)
        return table

    def _lookup(self,*dataset,clean=True,valid=False,train_weight=False):
        if len(dataset)==2:
            image,label = dataset
        elif len(dataset)==3:
            image,label,label2 = dataset
        else:
            raise ValueError("length of input dataset is incorrect")
        
        if clean:
            label = self.clean_table.lookup(label)

        if valid:
            if clean:
                weight = self.weight_table.lookup(label)
            else:
                weight = self.weight_table_full.lookup(label)
        elif train_weight:
            weight = self.train_weight_table.lookup(label)

        if len(dataset)==2:
            if valid or train_weight: 
                return image,label,weight
            return image,label
        else:
            if valid or train_weight: 
                return image,label,label2,weight
            return image,label,label2

    def lookup_clean_train(self,*dataset):
        return self._lookup(*dataset)
    
    def lookup_clean_valid(self,*dataset):
        return self._lookup(*dataset,valid=True)

    def lookup_full_train(self,*dataset):
        return self._lookup(*dataset,clean=False)

    def lookup_full_train_weight(self,*dataset):
        return self._lookup(*dataset,clean=False,train_weight=True)

    def lookup_full_valid(self,*dataset):
        return self._lookup(*dataset,clean=False,valid=True)

class Preprocess:
    def __init__(self,lookup,height,width,batch_size,valid_batch_size=None,test_batch_size=None):
        if not isinstance(lookup,Lookup):
            raise TypeError("lookup must be an instance of Lookup class")

        self.lookup = lookup
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size if valid_batch_size else batch_size
        self.test_batch_size = test_batch_size if test_batch_size else batch_size

    def _data_augment(self,*dataset,channels=3,seed=1214):
        if len(dataset)==2:
            image,label = dataset
        elif len(dataset)==3:
            image,label,label2 = dataset
        else:
            raise ValueError("length of input dataset is incorrect")

        p_spatial = tf.random.uniform([1], minval=0, maxval=1, dtype='float32', seed=seed)
        p_pixel = tf.random.uniform([1], minval=0, maxval=1, dtype='float32', seed=seed)
        p_crop = tf.random.uniform([1], minval=0, maxval=1, dtype='float32', seed=seed)
        
        ### Spatial-level transforms
        if p_spatial >= .2:
            image = tf.image.random_flip_left_right(image, seed=seed)
            
        if p_crop >= .7:
            if p_crop >= .95:
                image = tf.image.random_crop(image, size=[int(self.height*.6), int(self.width*.6), channels], seed=seed)
            elif p_crop >= .85:
                image = tf.image.random_crop(image, size=[int(self.height*.7), int(self.width*.7), channels], seed=seed)
            elif p_crop >= .8:
                image = tf.image.random_crop(image, size=[int(self.height*.8), int(self.width*.8), channels], seed=seed)
            else:
                image = tf.image.random_crop(image, size=[int(self.height*.9), int(self.width*.9), channels], seed=seed)
            image = tf.image.resize(image, size=[self.height, self.width])

            
        ## Pixel-level transforms
        if p_pixel >= .4:
            if p_pixel >= .85:
                image = tf.image.random_saturation(image, lower=0, upper=2, seed=seed)
            elif p_pixel >= .65:
                image = tf.image.random_contrast(image, lower=.8, upper=2, seed=seed)
            elif p_pixel >= .5:
                image = tf.image.random_brightness(image, max_delta=.2, seed=seed)
            else:
                image = tf.image.adjust_gamma(image, gamma=.6)

        if len(dataset)==2: return image, label
        return image,label,label2
        

    def _parse_image(self,image):
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0 
        image = tf.image.resize(image, (self.height, self.width))
        return image

    def _parse(self,example,aux=False,test=False):
        parsed_example = tf.io.parse_single_example(example, FORMAT)
        image = self._parse_image(parsed_example['encoded'])
        label =  tf.cast(parsed_example['label'], tf.int32) 
        if aux:
            label2 =  tf.cast(parsed_example['label2'], tf.int32)
            if test: return image, label2
            return image, label, label2
        return image, label

    def _parse_aux(self,example):
        return self._parse(example,aux=True)

    def _parse_test(self,example):
        return self._parse(example,aux=True,test=True)

    def _load_dataset(self,filenames,split="train",clean=True):
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        if split=="test":
            dataset = dataset.map(self._parse_test, num_parallel_calls=AUTO)
        else:
            if split=="train":
                dataset = dataset.with_options(ignore_order)
            if clean:
                dataset = dataset.map(self._parse, num_parallel_calls=AUTO)
            else:
                dataset = dataset.map(self._parse_aux, num_parallel_calls=AUTO)
        return dataset

    def get_dataset(self,file_pattern,clean=True,train_weight=False,
                    split="train",augment=True,shuffle_size=4096):
        if split not in ("train","test","valid"):
            raise ValueError("Split must be one of (train,test,valid)")

        filenames = tf.io.gfile.glob(file_pattern)
        if split=="train": 
            random.shuffle(filenames)
        dataset = self._load_dataset(filenames,split,clean)

        if split=="test":
            dataset = dataset.batch(self.test_batch_size)
        elif split=="train": 
            if augment:
                dataset = dataset.map(self._data_augment, num_parallel_calls=AUTO)
            dataset = dataset.repeat()
            if shuffle_size is not None:
                dataset = dataset.shuffle(shuffle_size)
            dataset = dataset.batch(self.batch_size) 

            if clean:
                dataset = dataset.map(self.lookup.lookup_clean_train, num_parallel_calls=AUTO)
            elif train_weight:
                dataset = dataset.map(self.lookup.lookup_full_train_weight, num_parallel_calls=AUTO)
            else:
                dataset = dataset.map(self.lookup.lookup_full_train, num_parallel_calls=AUTO)
        else: 
            dataset = dataset.batch(self.valid_batch_size)
            if clean:
                dataset = dataset.map(self.lookup.lookup_clean_valid, num_parallel_calls=AUTO)
            else:
                dataset = dataset.map(self.lookup.lookup_full_valid, num_parallel_calls=AUTO)
            # dataset = dataset.cache()

        dataset = dataset.prefetch(AUTO) 
        return dataset