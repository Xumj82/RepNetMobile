import tensorflow as tf
import numpy as np
import cv2
import pathlib
import pickle

def write_video(frames,fps,frame_width,frame_height,path = 'output.avi'):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(path, fourcc, fps, (frame_width,frame_height))
    for frame in frames:
        out.write(frame.numpy())
    out.release()

def get_frames(path):
    frames = []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frames.append(frame)
    
    cap.release()
    return frames, frame_width, frame_height, fps

def get_periodicity(y):   
    y = tf.round(y)-1
    y = tf.squeeze(y, axis=-1, name=None).numpy()
    periodicity_mitrixs  = tf.one_hot(y, 32,
           on_value=1.0, off_value=0.0,
           axis=-1)
    return periodicity_mitrixs

# 根据y计算回归矩阵，y=0 则为0， y>0 则为1
def get_with_in_period(y):
    y = tf.where(tf.math.greater(y, 1), 1, y)
    y = tf.cast(y, dtype=np.float32)
    return y

def get_preprocess_data(file):
    file = file.numpy()
    with open(file, 'rb') as f:
        x, y1, y2 = pickle.load(f)
    return x, y1, y2

def get_preprocess_data2(file):
    file = file.numpy()
    with open(file, 'rb') as f:
        x, y1, y2 = pickle.load(f)
        y1 = get_periodicity(y1)
    return x, y1, y2

def load_dataset(data_dir,sample_size,mode = 'All', onehot=True)->tf.data.Dataset:
    data_root = pathlib.Path(data_dir)
    if mode == 'combined':
        all_data_path = list(data_root.rglob('*countix_data*.pickle',))
    elif mode == 'synthetic':
        all_data_path = list(data_root.rglob('*synthetic_data*.pickle'))
    else:
        all_data_path = list(data_root.rglob('*.pickle'))

    all_data_path = [str(path) for path in all_data_path]
    #generate video (64*112*112*3)
    if onehot:
        dataset = tf.data.Dataset.from_tensor_slices(all_data_path).shuffle(sample_size, reshuffle_each_iteration=True).map(lambda x: tf.py_function(func=get_preprocess_data2,inp=[x],
        Tout=[tf.float32,tf.float32,tf.float32]))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(all_data_path).shuffle(sample_size, reshuffle_each_iteration=True).map(lambda x: tf.py_function(func=get_preprocess_data,inp=[x],
        Tout=[tf.float32,tf.float32,tf.float32]))
    return dataset

def load_dataset_test(data_dir,sample_size,mode = 'All', onehot=True)->tf.data.Dataset:
    data_root = pathlib.Path(data_dir)
    if mode == 'combined':
        all_data_path = list(data_root.glob('*countix_data*'))
    elif mode == 'synthetic':
        all_data_path = list(data_root.glob('*synthetic_data*'))
    else:
        all_data_path = list(data_root.glob('*'))

    all_data_path = [str(path) for path in all_data_path]
    #generate video (64*112*112*3)
    if onehot:
        dataset = tf.data.Dataset.from_tensor_slices(all_data_path).take(sample_size).shuffle(sample_size, reshuffle_each_iteration=True)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(all_data_path).take(sample_size).shuffle(sample_size, reshuffle_each_iteration=True)
    return dataset