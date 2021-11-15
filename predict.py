import imp
import os

import cv2
import numpy as np
import pandas as pd
import pathlib
# from scipy.signal import medfilt
from scipy.ndimage import median_filter as medfilt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import csv
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import dtype
from scipy.signal import find_peaks as findPeaks
from Datasets.SyntheticDataset import get_rep_video_mobilenet, get_with_in_period, view_test_vid

CHKPOINT_PATH = "ckpt/repnet-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(CHKPOINT_PATH)

def put_count_on_image(img, count,frame_width, frame_height):
    text = 'count:{:3.4f}'.format(count)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (0,frame_height)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1
    # Using cv2.putText() method
    image = cv2.putText(img, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    return image

def read_video2tensor(video_filename):
    """Read video from file."""
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frames = []
    frame_tensors = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_tensor = tf.cast(frame_bgr, tf.float32)
            frame_tensor = tf.image.per_image_standardization(frame_tensor)
            frame_tensor = tf.image.resize(frame_tensor, (112, 112))
            frame_tensors.append(frame_tensor)
            frames.append(frame_bgr)
    frame_tensors = tf.cast(frame_tensors, tf.float32)
    # frame_tensors = tf.expand_dims(frame_tensors, axis=0)
    cap.release()
    return frames, frame_tensors, fps, frame_width, frame_height

def read_video2tensor2(video_filename, model:ResnetPeriodEstimator):
    """Read video from file."""
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frames = []
    frame_tensors = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_tensors.append(frame_bgr)
            frames.append(frame_bgr)
    frame_tensors = model.preprocess(frame_tensors)
    # frame_tensors = tf.expand_dims(frame_tensors, axis=0)
    cap.release()
    return frames, frame_tensors, fps, frame_width, frame_height

def read_video2tensor3(video_filename):
    """Read video from file."""
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frames.append(frame_bgr)
    frames = np.asarray(frames, dtype=np.uint8)
    frame_tensors = tf.image.resize(frames,(112,112))
    # frame_tensors = tf.expand_dims(frame_tensors, axis=0)
    cap.release()
    return frames, frame_tensors, fps, frame_width, frame_height

def write_video(frames,fps,frame_width,frame_height,path = 'output.avi'):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(path, fourcc, fps, (frame_width,frame_height))
    for frame in frames:
        out.write(frame)
    out.release()

def write_video2(frame_tensors,fps,frame_width,frame_height,path = 'output.avi'):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(path, fourcc, fps, (frame_width,frame_height))
    for frame in frame_tensors:
      frame = tf.cast(frame, dtype=tf.uint8)
      out.write(frame.numpy())
    out.release()

def split2batch(frames:tf.Tensor,frame_width, frame_height,frame_tensors, model:Model):
    #视频帧数
    seq_len = frame_tensors.shape[0]
    #计算步长
    stride = round(seq_len/64)
    # stride = 8
    idxes = tf.range(0, seq_len, stride)
    idxes = tf.clip_by_value(idxes, 0, seq_len-1)
    pad_num = 64 - idxes.shape[0]
    idxes = np.pad(idxes.numpy(), (pad_num,0), mode='constant', constant_values=0)
    #压缩至 64 帧    
    curr_frames = tf.gather(frame_tensors, idxes)
    curr_frames = tf.expand_dims(curr_frames, axis=0)

    y1pred, y2pred = model.predict(curr_frames)

    y2pred  = tf.nn.sigmoid(y2pred)
    y1pred = tf.where(
      tf.math.less(y2pred, 0.5), 0.0, y1pred)
    y1pred = tf.where(
      tf.math.less(y1pred,2), 0.0, y1pred) 
    y1pred = tf.math.reciprocal_no_nan(y1pred, name=None)
    y1pred = tf.squeeze(y1pred, [0,-1]).numpy()
    
    f = 0
    period = []
    clip_range = []
    out_frames = []
    cur_count = 0
    while f < seq_len:     
        if f in idxes:
            itemindex = np.where(idxes==f)
            itemindex = itemindex[0][0]
            period.append(y1pred[itemindex])
            in_period = y2pred[0][itemindex]
            if in_period < 0.5 or f == 0:
                range = []
                clip_range.append(range)
            else:
                range = clip_range[-1]
                range.append(f)
        elif len(clip_range[-1]) > 0:
            clip_range[-1].append(f)

        count = np.sum(period)
        if int(count) - cur_count == 1:
            range = []
            clip_range.append(range)
            cur_count = int(count)
        frame_out = put_count_on_image(frames[f],np.sum(period),frame_width, frame_height)
        out_frames.append(frame_out)
        f += 1
    return out_frames

def get_sim_imgs(frame_tensors:tf.Tensor,stride, model:Model ,batch_size,constant_speed=False,num_frames=64, image_size = 112):
    imgs = []
    #视频帧数
    seq_len = frame_tensors.shape[0]
    #计算步长
    num_batches = int(np.ceil(seq_len/num_frames/stride/batch_size))
    w = 64
    h = 64
    fig = plt.figure(figsize=(64, 64))
    columns = num_batches
    rows = 1
    for batch_idx in range(num_batches):
        idxes = tf.range(batch_idx*batch_size*num_frames*stride,
            (batch_idx+1)*batch_size*num_frames*stride,
            stride)
        idxes = tf.clip_by_value(idxes, 0, seq_len-1) #压缩 idex 使其最大值不超过视频长度
        #压缩至 64 帧    
        curr_frames = tf.gather(frame_tensors, idxes)
        curr_frames = tf.reshape(curr_frames,[batch_size, num_frames, image_size, image_size, 3])

        sim = model.predict(curr_frames)

        sim = np.reshape(sim, (64,64))
        sim = np.array(sim * 255, dtype = np.uint8)

        sim = cv2.equalizeHist(sim)

        plt.title('Stride:{0}  batch:{1}'.format(stride, batch_idx), fontsize=8)
        max_idx = tf.math.reduce_max(idxes, axis=None, keepdims=False, name=None)
        min_idx = tf.math.reduce_min(idxes, axis=None, keepdims=False, name=None)
        
        fig.add_subplot(rows, columns, batch_idx+1)
        plt.imshow(sim,extent=[min_idx,max_idx,min_idx,max_idx])
        # imgplot = plt.imshow(sim*255,extent=[min_idx,max_idx,min_idx,max_idx])
        # plt.show()
        # imgs.append(sim)
    plt.show()
    plt.savefig('assert/output_sim.png')

def generate_video_with_counts(frames,fps,frame_width, frame_height, per_frame_counts, path = './data/output.avi'):   
    seq_len = len(frames)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    f = 0
    while f < seq_len:     
        count = np.sum(per_frame_counts[:f])
        frame_out = put_count_on_image(frames[f],count,frame_width, frame_height)
        out.write(frame_out)
        f += 1
    out.release()
    return count

def get_score(period_score, within_period_score, stride, length): #period_score: 每一帧是哪一个 周期（1，2...32）的概率， within_period_score:是不是周期的概率
  """Combine the period and periodicity scores."""
  within_period_score = tf.nn.sigmoid(within_period_score)[:, 0] #64*1 array applied sigmod and convert it to a 1D array between (0,1)
  per_frame_periods = tf.argmax(period_score, axis=-1) + 1 #Returns the index with the largest value across axes of a tensor.
  pred_period_conf = tf.reduce_max(
     period_score, axis=-1) #softmax of the 64*32 array, then get the max of every 32 array and output a 1D 64 array as confidence of per_frame_periods
  pred_period_conf = tf.where(
      tf.math.less(per_frame_periods, 3), 0.0, pred_period_conf) #周期小于3 则认为是假周期， conf设为0
  within_period_score *= pred_period_conf #某周期的概率 * 是否为周期的概率
  within_period_score = np.sqrt(within_period_score)
  pred_score = tf.reduce_mean(within_period_score)
  return pred_score, within_period_score

def get_counts(model, frames, strides, batch_size,
               threshold,
               within_period_threshold,
               num_frames=64,
               image_size = 112,
               constant_speed=True,
               median_filter=True,
               fully_periodic=False):
  """Pass frames through model and conver period predictions to count."""
  seq_len = frames.shape[0]
  raw_scores_list = []
  scores = []
  within_period_scores_list = []

  if fully_periodic:
    within_period_threshold = 0.0


  for stride in strides:
    num_batches = int(np.ceil(seq_len/num_frames/stride/batch_size)) #（seq_len/model.num_frames/stride/batch_size）取整
    raw_scores_per_stride = []
    within_period_score_stride = []
    for batch_idx in range(num_batches):
      idxes = tf.range(batch_idx*batch_size*num_frames*stride,
                       (batch_idx+1)*batch_size*num_frames*stride,
                       stride)
      idxes = tf.clip_by_value(idxes, 0, seq_len-1) #压缩 idex 使其最大值不超过视频长度
      curr_frames = tf.gather(frames, idxes) #按 idex 顺序扩展原视频
      curr_frames = tf.reshape(
          curr_frames,
          [batch_size, num_frames, image_size, image_size, 3]) #将原视频分为 20 个batch

      raw_scores, within_period_scores = model.predict(curr_frames) # raw_scores: 20个batch的 plength 64*32,   within_period_scores: 20个batch的periodicity
      raw_scores_per_stride.append(np.reshape(raw_scores,
                                              [-1, num_frames//2]))
      within_period_score_stride.append(np.reshape(within_period_scores,
                                                   [-1, 1]))
    raw_scores_per_stride = np.concatenate(raw_scores_per_stride, axis=0)
    raw_scores_list.append(raw_scores_per_stride)
    within_period_score_stride = np.concatenate(
        within_period_score_stride, axis=0)
    pred_score, within_period_score_stride = get_score(
        raw_scores_per_stride, within_period_score_stride, stride=stride, length=seq_len)
    scores.append(pred_score)
    within_period_scores_list.append(within_period_score_stride)
  
  
  # if scores[0]<0.85 and len(scores)>1:
  #   import pandas as pd
  #   scores_pd = pd.DataFrame(scores)[0]
  #   (prmPks ,_) = findPeaks (scores_pd,prominence=0.02)
  #   if len(prmPks) >0:
  #     argmax_strides = prmPks[0]
  #   else:
  #     argmax_strides = np.argmax(scores)
  #   # plt.plot(scores_pd)
  #   # plt.plot(prmPks,scores_pd[prmPks],'x')
  #   # plt.show()
  # else:
  #   # Stride chooser
  #   argmax_strides = 0 #选择均分最高的 stride
  argmax_strides = np.argmax(scores)
  chosen_stride = strides[argmax_strides] #stride index
  
  raw_scores = np.repeat(
      raw_scores_list[argmax_strides], chosen_stride, axis=0)[:seq_len]
  within_period = np.repeat(
      within_period_scores_list[argmax_strides], chosen_stride,
      axis=0)[:seq_len]
  within_period_binary = np.asarray(within_period > within_period_threshold)
  if median_filter:
    within_period_binary = medfilt(within_period_binary, 5)

  # Select Periodic frames
  periodic_idxes = np.where(within_period_binary)[0] #选择 为true的frame 的 index

  if constant_speed:
    # Count by averaging predictions. Smoother but
    # assumes constant speed.
    scores = tf.reduce_mean(raw_scores[periodic_idxes], axis=0) #每周期得分 32 length array
    max_period = np.argmax(scores) #最高分 周期
    pred_score = scores[max_period]
    pred_period = chosen_stride * (max_period + 1)
    per_frame_counts = (
        np.asarray(seq_len * [1. / pred_period]) *
        np.asarray(within_period_binary))
  else:
    # Count each frame. More noisy but adapts to changes in speed.
    pred_score = tf.reduce_mean(within_period)
    per_frame_periods = tf.argmax(raw_scores, axis=-1) + 1
    per_frame_counts = tf.where(
        tf.math.less(per_frame_periods, 3),
        0.0,
        tf.math.divide(1.0,
                       tf.cast(chosen_stride * per_frame_periods, tf.float32)),
    )
    if median_filter:
      per_frame_counts = medfilt(per_frame_counts, 5)

    per_frame_counts *= np.asarray(within_period_binary)

    pred_period = seq_len/np.sum(per_frame_counts)

  if pred_score < threshold:
    print('No repetitions detected in video as score '
          '%0.2f is less than threshold %0.2f.'%(pred_score, threshold))
    per_frame_counts = np.asarray(len(per_frame_counts) * [0.])

  return (pred_period, pred_score, within_period,
          per_frame_counts, chosen_stride)

def split_video_to_single_act(frames, per_frame_counts, pred_period):
  f = 0
  count = 0
  clip_range = [[]]
  clip_frames = []
  while f < len(per_frame_counts):
    clip_range[-1].append(f)
    if int(count+per_frame_counts[f])-int(count) == 1:
      clip_range.append([f])
    count += per_frame_counts[f]
    f += 1
  for idxes in clip_range:
    if len(idxes) > pred_period/2:
      new_frames = np.take(frames, idxes,axis=0)
      clip_frames.append(new_frames)
  return clip_frames


def predict_from_mbl_mod(video_path,batch_size=5,output_path='G:/RepNet/data/output.avi'):
  checkpoint_dir = "ckpt/"
  model = rep_net_mobile(batch_size)
  sim_layer = rep_net_mobile_sim(1)
  latest = tf.train.latest_checkpoint(checkpoint_dir)
  if latest:
      model.load_weights(latest)
      sim_layer.load_weights(latest)
  frames, frame_tensors, fps, frame_width, frame_height = read_video2tensor3(video_path)
  
  # write_video2(tf.cast(frame_tensors,tf.uint8),fps,112,112,output_path)
  strides = np.arange(1, 8, dtype=int)
  print(strides)
  (pred_period, pred_score, within_period,per_frame_counts, chosen_stride) = get_counts(
      model=model,
      frames=frame_tensors, 
      strides=strides,
      batch_size=batch_size, 
      threshold=0.2, 
      image_size=112,
      within_period_threshold=0.5,
      constant_speed=False,
      median_filter=True
      )
  print(chosen_stride)
  # count = generate_video_with_counts(frames,fps,frame_width, frame_height, per_frame_counts, path = output_path)
  get_sim_imgs(frame_tensors,chosen_stride,sim_layer,1)

video_path = "G:/RepNet/data/725.mp4" #"./data/QUVARepetitionDataset/videos/002_swing_forward.mp4"
# video_path = "./data/QUVARepetitionDataset/videos/002_swing_forward.mp4"
predict_from_mbl_mod(video_path)
# predict_from_google_mod(video_path)

# with open('G:/RepNet/data/testdata/p.csv', 'w', newline='', encoding='utf-8') as f:
#   writer = csv.writer(f)
#   for i in range(20):
#     x, y1,y2,y = view_test_vid('G:/RepNet/data/synthvids/train24.mp4',output='G:/RepNet/data/testdata/video{0}.avi'.format(i))
#     y = tf.squeeze(y, -1)
#     writer.writerow(y.numpy())
# write_video2(x,10,112,112)
# print("test")