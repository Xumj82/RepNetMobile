import os
import cv2
import csv
import glob
import pickle
import random
import functools
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from numpy.core.numeric import outer
from multiprocessing import Pool, cpu_count
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_math_ops import mod
from Datasets.utility import get_periodicity, get_frames, get_with_in_period

def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def compress_idxs(length, output_length):
  delta = length//output_length
  if delta>0:
    new_idxs = tf.range(length,delta=delta).numpy()[:output_length]
  else:
    new_idxs = tf.cast([],dtype=tf.int32)
  return new_idxs

def get_durs(end_start_weight,repeat_dur,output_frames_length):
  end_start = np.random.choice(['end', 'start', 'middle'], p = end_start_weight)
  if end_start == 'end':
    start_dur = output_frames_length-repeat_dur
    end_dur = 0
  if end_start == 'start':
    end_dur = output_frames_length-repeat_dur
    start_dur = 0
  if end_start == 'middle':
    start_dur = start_dur = random.randint(0, output_frames_length-repeat_dur)
    end_dur = output_frames_length - start_dur-repeat_dur
  return start_dur, end_dur

def random_noise(img):
    '''
    img is a 3-D variable (ex: tf.Variable(image, validate_shape=False) ) and  HWC order
    '''
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.cast(img, dtype=tf.float32)
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.5, dtype=tf.float32)
    img = tf.add(img, noise)
    return img

def generate_rep_vid(frames,reversal=False, mode='cut',output_frames_length=64,end_start_weight=[0.4,0.4,0.2], static_nature_weight=[0.8,0.2], min_period = 3):
  
  if mode=='cut':
    even_period = np.random.choice([True, False], p = [0.5,0.5])
    if reversal:
      if even_period:
        half_repeat_period = random.randint(min_period,output_frames_length // 4)
        repeat_period = half_repeat_period*2-1
        count = (output_frames_length-1) // (repeat_period-1)
        repeat_dur = (repeat_period-1) * count+1
        start_dur = random.randint(0, output_frames_length-repeat_dur)
        end_dur = output_frames_length - start_dur-repeat_dur
        repeat_idxs = compress_idxs(len(frames), half_repeat_period)
        for i in range(count*2-1):
          repeat_idxs = np.pad(repeat_idxs,[0,half_repeat_period-1], "reflect")
        start_idxs = np.flip(np.flip(repeat_idxs)[1:start_dur+1])
        end_idxs = repeat_idxs[1:end_dur+1]
        repeat_period = repeat_period -1
      else:
        half_repeat_period = random.randint(min_period,output_frames_length // 4)
        repeat_period = half_repeat_period*2-1
        count = (output_frames_length) // (repeat_period)
        repeat_dur = repeat_period * count
        start_dur = random.randint(0, output_frames_length-repeat_dur)
        end_dur = output_frames_length - start_dur-repeat_dur
        half_repeat_idxs = compress_idxs(len(frames), half_repeat_period)
        repeat_idxs = np.pad(half_repeat_idxs,[0,half_repeat_period-1], "reflect")
        repeat_idxs = tf.expand_dims(repeat_idxs, 0)
        repeat_idxs = tf.repeat(repeat_idxs, repeats=count, axis=0)
        repeat_idxs = tf.reshape(repeat_idxs,[repeat_idxs.shape[0]*repeat_idxs.shape[1]])
        start_idxs = np.flip(np.flip(repeat_idxs)[0:start_dur])
        end_idxs = repeat_idxs[0:end_dur]
        repeat_period = repeat_period
    else:
      repeat_period = random.randint(min_period,output_frames_length // 2)
      count = output_frames_length // repeat_period
      repeat_dur = repeat_period * count
      start_dur = random.randint(0,output_frames_length % repeat_period)
      end_dur = output_frames_length % repeat_period -start_dur
      
      repeat_idxs = compress_idxs(len(frames), repeat_period)
      repeat_idxs = tf.expand_dims(repeat_idxs, 0)
      repeat_idxs = tf.repeat(repeat_idxs, repeats=count, axis=0)
      repeat_idxs = tf.reshape(repeat_idxs,[repeat_idxs.shape[0]*repeat_idxs.shape[1]])

      start_idxs = np.flip(np.flip(repeat_idxs)[0:start_dur])
      end_idxs = repeat_idxs[0:end_dur]

    y = np.full((output_frames_length),repeat_period)
  if mode=='norepeat':
    # start = random.randint(0, len(frames)-ou)
    start_idxs = compress_idxs(len(frames), output_frames_length)
    repeat_idxs = tf.constant([],dtype=tf.int32)
    end_idxs =  tf.constant([],dtype=tf.int32)
    y = np.full((output_frames_length), 0)
  if mode=='nature':
    static_pic = np.random.choice([True, False], p = static_nature_weight)
    even_period = np.random.choice([True, False], p = [0.5,0.5])
    if reversal:
      if even_period:
        half_repeat_period = random.randint(min_period,output_frames_length // 5)
        repeat_period = half_repeat_period*2-1
        count = random.randint(2, (output_frames_length-half_repeat_period)//repeat_period)
        repeat_dur = (repeat_period-1) * count+half_repeat_period

        start_dur, end_dur = get_durs(end_start_weight, repeat_dur, output_frames_length)

        total_idxs = compress_idxs(len(frames), start_dur+half_repeat_period+end_dur)

        repeat_idxs = total_idxs[start_dur:half_repeat_period+start_dur]      
        for i in range(count*2):
          repeat_idxs = np.pad(repeat_idxs,[0,half_repeat_period-1], "reflect")

        if static_pic:
          start_idxs = tf.constant(total_idxs[start_dur-1], shape=(start_dur))
          end_idxs = tf.constant(total_idxs[half_repeat_period+start_dur-1], shape=end_dur)
        else:
          start_idxs = total_idxs[0:start_dur]
          end_idxs = total_idxs[half_repeat_period+start_dur:]

        repeat_period = repeat_period -1
      else:
        half_repeat_period = random.randint(min_period,output_frames_length // 5)
        repeat_period = half_repeat_period*2-1
        count = random.randint(2, (output_frames_length-half_repeat_period)//repeat_period)
        repeat_dur = repeat_period * count+half_repeat_period

        start_dur, end_dur = get_durs(end_start_weight, repeat_dur, output_frames_length)

        total_idxs = compress_idxs(len(frames), start_dur+half_repeat_period+end_dur)
        
        half_repeat_idxs = total_idxs[start_dur:half_repeat_period+start_dur]
        repeat_idxs = np.pad(half_repeat_idxs,[0,half_repeat_period-1], "reflect")
        repeat_idxs = tf.expand_dims(repeat_idxs, 0)      
        repeat_idxs = tf.repeat(repeat_idxs, repeats=count, axis=0)
        repeat_idxs = tf.reshape(repeat_idxs,[repeat_idxs.shape[0]*repeat_idxs.shape[1]])
        repeat_idxs = tf.concat([repeat_idxs, half_repeat_idxs],0)
        if static_pic:
          start_idxs = tf.constant(total_idxs[start_dur-1], shape=(start_dur))
          end_idxs = tf.constant(total_idxs[half_repeat_period+start_dur-1], shape=end_dur)
        else:
          start_idxs = total_idxs[0:start_dur]
          end_idxs = total_idxs[half_repeat_period+start_dur:]

        repeat_period = repeat_period    
    else:
      repeat_period = random.randint(min_period,output_frames_length // 2)
      count = random.randint(2, output_frames_length//repeat_period)
      repeat_dur = repeat_period * count

      start_dur, end_dur = get_durs(end_start_weight, repeat_dur, output_frames_length)
      
      total_idxs = compress_idxs(len(frames), start_dur+repeat_period+end_dur)

      repeat_idxs = total_idxs[start_dur:repeat_period+start_dur]    
      repeat_idxs = tf.expand_dims(repeat_idxs, 0)
      repeat_idxs = tf.repeat(repeat_idxs, repeats=count, axis=0)
      repeat_idxs = tf.reshape(repeat_idxs,[repeat_idxs.shape[0]*repeat_idxs.shape[1]])

      if static_pic:
        start_idxs = tf.constant(total_idxs[start_dur-1], shape=(start_dur))
        end_idxs = tf.constant(total_idxs[repeat_period+start_dur-1], shape=end_dur)
      else:
        start_idxs = total_idxs[0:start_dur]
        end_idxs = total_idxs[repeat_period+start_dur:]

    y = np.full((start_dur), 0)
    y = np.concatenate((y,np.full((repeat_dur),repeat_period)),0)
    y = np.concatenate((y,np.full((end_dur),0)),0)
  
  total_idxs = tf.concat([start_idxs, repeat_idxs, end_idxs], 0)
  frames = np.asarray(frames, dtype=np.uint8)
  output = tf.gather(frames, total_idxs).numpy()
  assert total_idxs.shape[0] == output_frames_length
  y = tf.expand_dims(y, axis = -1)
  y1 = get_periodicity(y)
  y2 = get_with_in_period(y)
  # output = frames
  return output, y1, y2, y

def rotation(frame,delta_degree,param):
    return tf.cast(tf.keras.preprocessing.image.apply_affine_transform(frame,theta=delta_degree*param,fill_mode='constant'),dtype=tf.float32)

def vertical_tran(frame,delta_tran,param):
    return tf.cast(tf.keras.preprocessing.image.apply_affine_transform(frame,tx=delta_tran*param,fill_mode='constant'),dtype=tf.float32)

def horizon_tran(frame,delta_tran,param):
    return tf.cast(tf.keras.preprocessing.image.apply_affine_transform(frame,ty=delta_tran*param,fill_mode='constant'),dtype=tf.float32)

def scale(frame,delta_zoom,param):
    return tf.cast(tf.keras.preprocessing.image.apply_affine_transform(frame,zx=delta_zoom*param+1,zy=delta_zoom*param+1,fill_mode='constant'),dtype=tf.float32)

def load_image(filename):
    raw = tf.io.read_file(filename)
    image = tf.image.decode_png(raw, channels=3)
    return image

def get_mask(path):
    idx = random.randint(1,201)
    all_mask_files = glob.glob(path+"{0}/seg*.png".format(str(idx)))
    all_masks = []
    for mask_file in all_mask_files:
        mask = load_image(mask_file)
        all_masks.append(mask)
    return all_masks

def op_tf(mid_pos,frame,seq,length,tran_type, delta_degree,delta_tran, delta_zoom):
    param = tf.cast(tf.cond(tf.less_equal(seq,mid_pos), lambda:seq*1/mid_pos, lambda:1-2*(seq-mid_pos)/(length-mid_pos)), dtype=tf.float32)
    frame = tf.case(
        [(tf.math.equal(tran_type, 'rotation'), lambda:rotation(frame,delta_degree,param)), 
        (tf.math.equal(tran_type, 'vertical_tran'), lambda:vertical_tran(frame,delta_tran,param)),
        (tf.math.equal(tran_type, 'horizon_tran'), lambda:horizon_tran(frame,delta_tran,param)),
        (tf.math.equal(tran_type, 'scale'), lambda:scale(frame,delta_zoom,param)),
                ],
        default=lambda:rotation(frame,delta_degree,param), exclusive=True)
    # img = tf.image.random_brightness(img, max_delta=0.2)
    noise = tf.random.normal(shape=tf.shape(frame), mean=0, stddev=0.5, dtype=tf.float32)
    frame = tf.add(frame, noise)
    return frame

def random_transform(images,tran_type,delta_degree,delta_tran, delta_zoom):
  mid_pos = tf.random.uniform(shape=(), minval=10, maxval=tf.shape(images)[0]-10, dtype=tf.int32)
  # idxes = tf.range(tf.shape(images)[0], dtype=tf.int32)
  # length_row = tf.constant(tf.shape(images)[0], shape=(tf.shape(images)[0]), dtype=tf.int32)
  # tran_type_row = tf.constant(tran_type, shape=(tf.shape(images)[0]), dtype=tf.string)
  # delta_degree_row = tf.constant(delta_degree, shape=(tf.shape(images)[0]), dtype=tf.float32)
  # delta_tran_row = tf.constant(delta_tran, shape=(tf.shape(images)[0]), dtype=tf.float32)
  # delta_zoom_row = tf.constant(delta_zoom, shape=(tf.shape(images)[0]), dtype=tf.float32)
  
  seq = 0
  
  while seq < len(images):
      images[seq] = op_tf(mid_pos,images[seq],seq,tf.shape(images)[0],tran_type,delta_degree,delta_tran,delta_zoom)
      seq +=1
  return images

def write_video(frames,fps,frame_width,frame_height,path = 'output.avi'):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(path, fourcc, fps, (frame_width,frame_height))
    for frame in frames:
        frame = tf.cast(frame, dtype=tf.uint8)
        out.write(frame.numpy())
    out.release()

class SyntheticDataGenerator():
  def __init__(self, video_path,maskpath, save_path, sample_size, output_size=(224,224), start_idx = 0, cpu_num=6,mask_layer_num =3):
    self.video_path = video_path
    self.maskpath =maskpath 
    self.save_path = save_path
    self.sample_size = sample_size
    self.start_idx = start_idx
    self.cpu_num = cpu_num
    self.output_size = output_size
    self.mask_layer_num = mask_layer_num

  def generate(self):
      arr = list(range(self.start_idx,self.sample_size+self.start_idx))
      with Pool(self.cpu_num) as p:
        r = list(tqdm(p.imap(self.save_rep_video, arr), total=len(arr)))

  def view_test_vid(self,output=None):
    # path = path.decode('utf-8')
    frames, y1, y2, y = self.get_rep_video()
    frames = tf.cast(frames,dtype=tf.uint8)
    if output:
      write_video(frames,1,self.output_size[0],self.output_size[1],path = output)
      print(tf.argmax(y1, axis=-1))
      print(tf.squeeze(y2,axis=-1))

  def view_pickle_vid(self,pickle_path,output=None):
    # path = path.decode('utf-8')
    with open(pickle_path, 'rb') as f:
        frames, y1, y2 = pickle.load(f)
    frames = tf.cast(frames,dtype=tf.uint8)
    if output:
      write_video(frames,1,self.output_size[0],self.output_size[1],path = output)
      print(tf.argmax(y1, axis=-1))
      print(tf.squeeze(y2,axis=-1))

  def save_rep_video(self,idx):
    # path = path.decode('utf-8')
    frames, y1, y2, y = self.get_rep_video()
    save_path = formatter(self.save_path, '*_synthetic_data.pickle', idx)
    with open(save_path, 'wb') as f:
      pickle.dump((frames, y1, y2), f)
    
  def get_rep_video(self):
      with tf.device('/CPU:0'):
        path =os.path.join(self.video_path,'train*.mp4')
        path = random.choice(glob.glob(path))

        frames, frame_width, frame_height, fps = get_frames(path)
        assert len(frames)>64, "Video less than 64: {0} ".format(path)
        reversal = np.random.choice([True, False], p = [0.8, 0.2])
        mode = np.random.choice(['cut', 'nature','norepeat'], p = [0.5,0.4,0.1])
        frames, y1, y2, y = generate_rep_vid(frames,reversal=reversal, mode=mode)

        #random transform
        delta_degree = random.randint(15,20)
        delta_tran=random.randint(15,20)
        delta_zoom=random.uniform(0.1, 0.4)
        tran_type = np.random.choice(['rotation', 'vertical_tran', 'horizon_tran','scale'], p = [0.25, 0.25, 0.25,0.25])
        frames = random_transform(frames,tran_type,delta_degree=delta_degree, delta_tran=delta_tran, delta_zoom=delta_zoom)
        frames = tf.cast(tf.image.resize(frames, self.output_size),dtype=tf.uint8)

        for i in range(0,self.mask_layer_num):
          masks = get_mask(self.maskpath)
          masks = tf.cast(tf.image.resize(masks, self.output_size),dtype=tf.uint8)
          frames = tf.bitwise.bitwise_or(masks,frames)
        frames = tf.cast(frames,dtype=tf.float32)

        assert frames.shape[0]==64, "Video length wrong "
        assert frames.dtype==tf.float32, "Video in wrong format: {0} ".format(str(frames.dtype))
      return frames, y1, y2, y
