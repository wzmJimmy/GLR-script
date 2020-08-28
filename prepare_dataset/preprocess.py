# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

# from google.colab import auth
# auth.authenticate_user()

import requests
def add_client_retries(client,max_retries=5):
    """ Retry connection to prevent connectionResetError
    """
    adapter = requests.adapters.HTTPAdapter(max_retries=max_retries)
    client._http.mount("https://", adapter)
    client._http._auth_request.session.mount("https://", adapter)
    return client

GCS_PATH = "gs://gld_v2"
PROJECT_ID = 'rational-moon-286222'
from google.cloud import storage
# storage_client = storage.Client(project=PROJECT_ID)

def upload_blob( source_file_name, destination_blob_name,bucket_name='jimmy_colab',verb=True):
    storage_client = add_client_retries(storage.Client(project=PROJECT_ID))
    
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    if verb: print('File {} uploaded to {}.'.format(source_file_name,destination_blob_name))

def download_blob( source_blob_name, destination_file_name,bucket_name='gld_v2',verb=True):
    storage_client = add_client_retries(storage.Client(project=PROJECT_ID))
    
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    if verb: print('File {} downloaded from {}.'.format(destination_file_name,source_blob_name))

"""# Dowload and Load"""

file = ["df_clean_tr.csv","df_clean_va.csv","df_out_tr.csv","df_out_va.csv","df_test.csv"]
if not os.path.exists(file[0]):
    for i in file:
        download_blob(i,i)

# 1580470/100,2552444/200,117577/8
# (15804.7, 12762.22, 14697.125)

# 1392698/15804, 187772/15804, 2252318/12762, 300126/12762
# (88.12313338395343, 11.881295874462161, 176.48628741576556, 23.517160319699105)
# 88,12,177,24

"""# Process"""

def _process_image(filename):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()
    image = tf.io.decode_jpeg(image_data, channels=3)
    
    if len(image.shape) != 3:
        raise ValueError('The parsed image number of dimensions is not 3 but %d' % (image.shape))
    if image.shape[2] != 3:
        raise ValueError('The parsed image channels is not 3 but %d' %(image.shape[2]))
    
    return image_data

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(file_id, image_buffer, label2, label=None):
    features = {
      'id': _bytes_feature(file_id.encode('utf-8')),
      'encoded': _bytes_feature(image_buffer),
      'label2': _int64_feature(label2)
      }
    if label is not None: features['label'] = _int64_feature(label)

    example = tf.train.Example(features=tf.train.Features(feature=features))
    
    return example

def find_path(idx):
    return "{}/{}/{}/{}.jpg".format(idx[0],idx[1],idx[2],idx)

def unit_map(row,folder,ind):
    image_path = find_path(row["id"])
    image_path = "{}/{}/{}".format(GCS_PATH,folder,image_path) 
    image_buffer = _process_image(image_path)

    label,label2 = None,True
    if ind != "test": label = row["landmark_id"]
    if ind != "clean": label2 = row["has_clean_landmark"]

    example = _convert_to_example(row["id"], image_buffer,label2, label)
    return example.SerializeToString()

pbar_batch = 100
def _write_tfrecord(shard,df,split,ind,in_folder,out_folder):
    output_file = '%s-%.5d.tfrec' % (split,shard)
    writer = tf.io.TFRecordWriter(output_file)
    print('Processing shard ', shard, ' and writing file ', output_file)
        
    count,pbar = 0,tqdm(total=df.shape[0])
    for _,row in df.iterrows():
        writer.write(unit_map(row,in_folder,ind))

        count += 1
        if count%pbar_batch==0:
            count = 0
            pbar.update(pbar_batch)
    pbar.update(count)
    pbar.close()
    writer.close()

    upload_blob(output_file, out_folder + output_file)
    os.remove(output_file)


def build_tfrecord_dataset(csv_path,num_shard,split,ind,in_folder,out_folder,nproc=16,gather=None):
    df = pd.read_csv(csv_path)
    # df = df.iloc[:500]
    li_df = list(enumerate(np.array_split(df,num_shard)))
    if gather is not None: 
        li_df = [li_df[i] for i in gather]
        
    write_tfrecord = partial(_write_tfrecord,split=split,in_folder=in_folder,
                             out_folder=out_folder,ind=ind)

    print("Start processing %s\n"%csv_path)
    
    with Pool(processes=nproc) as pool:
        pool.starmap(write_tfrecord, li_df)
        
    print("Finish processing %s\n"%csv_path)

    
if __name__=="__main__":
    
    file = ["df_clean_tr.csv","df_clean_va.csv","df_out_tr.csv","df_out_va.csv","df_test.csv"]
    shards = [88,12,177,24,8]
    # shards = [4,4,4,4,4]
    ind = ["clean","clean","out","out","test"]
    split = ["train","valid","train","valid","test"]
    in_folder = ["train","train","train","train","test"]
    out_folder = ["train_cleaned/train/","train_cleaned/valid/","train_outside/train/","train_outside/valid/","test/"]

    
    for i in range(5):
        build_tfrecord_dataset(file[i],shards[i],split[i],ind[i],in_folder[i],out_folder[i])