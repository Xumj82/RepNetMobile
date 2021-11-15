import pickle

import numpy as np
import pandas as pd
import pathlib
from multiprocessing import Pool
import cv2
import glob
import random
from random import randint
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import concat
from tqdm import tqdm
import os
from Datasets.utility import get_periodicity, get_with_in_period, get_frames, write_video

def get_frames(path):
    frames = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frames.append(frame)
    
    cap.release()
    
    return frames

def get_countix_dataset(path):
    df = pd.read_csv(path)
    return df

def get_combined_video(video_path,countix_file,output_len = 64 ):
    countix = pd.read_csv(countix_file)
    curFrames = []
    while len(curFrames)< 64:
        idx = randint(0, countix.shape[0])
        path = video_path+'/video_{0}.mp4'.format(countix.iloc[idx]['idx'])
        count = countix.iloc[idx]['count']
        if os.path.exists(path):
            curFrames = get_frames(path)

    mode = np.random.choice(['start', 'end','constant'], p = [0.3,0.3,0.4])        
    newFrames = []
    start_dur = 0
    end_dur = 0
    if mode == 'start':
        output_len = min(len(curFrames), randint(44, 60))
        start_dur = 64 - output_len
        for i in range(1, output_len + 1):
            newFrames.append(curFrames[i * len(curFrames)//output_len  - 1])
        finalFrames = [newFrames[0] for i in range(start_dur)]
        finalFrames.extend(newFrames)

    if mode == 'end':
        output_len = min(len(curFrames), randint(44, 64))
        end_dur = 64 - output_len
        for i in range(1, output_len + 1):
            newFrames.append(curFrames[i * len(curFrames)//output_len  - 1])
        finalFrames = newFrames
        finalFrames.extend([newFrames[-1] for i in range(end_dur)])

    if mode == 'constant':
        for i in range(1, output_len + 1):
            newFrames.append(curFrames[i * len(curFrames)//output_len  - 1])
        finalFrames = newFrames

    imgs = []
    for img in finalFrames:
        img_tensor = tf.cast(img, tf.float32)
        img_tensor = tf.image.resize(img_tensor, [112, 112])
        imgs.append(img_tensor)
    imgs = tf.cast(imgs, tf.float32)

    y = [0 for i in range(0,start_dur)]
    y.extend([output_len/count if 1<output_len/count<=32 else 0 for i in range(0, output_len)])
    y.extend( [ 0 for i in range(0, end_dur)] )
    y = tf.expand_dims(y, -1)
    y1 = get_periodicity(y)
    
    y2 = get_with_in_period(y)

    return imgs, y1, y2

# contix labeled repetition videos(64 frames)
def combined_data_preprocess(video_path, countix_file,sample_size, save_path ,start_idx = 0):
    #get path collection of videos
    #get count number from csv file
    i = start_idx
    # with Pool(6) as p:
    #   r = list(tqdm.tqdm(p.imap(get_combined_video,path,count), total=countix.shape[0]))
    with tqdm(total=sample_size) as pbar:
        while i <sample_size+start_idx:
            data = get_combined_video(video_path, countix_file)
            with open(save_path+'countix_data_{0}.pickle'.format(i), 'wb') as f:
                pickle.dump(data, f)
            pbar.update(1)
            i += 1

def view_test_vid(video_path, countix_file, save_path):
    # with Pool(6) as p:
    #   r = list(tqdm.tqdm(p.imap(get_combined_video,path,count), total=countix.shape[0]))
    frames, y1, y2 = get_combined_video(video_path, countix_file)
    if save_path:
      write_video(tf.cast(frames,dtype=tf.uint8),1,112,112,path = save_path)
      print(tf.argmax(y1, axis=-1))
      print(tf.squeeze(y2,axis=-1))


