import youtube_dl
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time, random
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
def download_video_from_url(idx,start_time, end_time,url_to_video,path_to_video='G:/RepNet/data/trainvids/'):
    dl_path = path_to_video+'video_dl_{0}.mp4'.format(idx)
    final_path = path_to_video+'video_{0}.mp4'.format(idx)
    if os.path.exists(dl_path):
        os.remove(dl_path)
    if os.path.exists(final_path):
        os.remove(final_path)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]',
        'outtmpl': str(dl_path),
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_to_video])
        
        ffmpeg_extract_subclip(dl_path , start_time, end_time, targetname=final_path)
        if os.path.exists(dl_path):
            os.remove(dl_path)
    except Exception as e: # work on python 2.x
        print('video_{0} : {1}'.format(idx, str(e)))
  
if __name__=='__main__':
    countix_path = './countix/countix_train.csv'
    video_dir = 'G:/RepNet/data/trainvids/'
    csv_path = 'G:/RepNet/data/trainvids/countix_df.csv'
    countix = pd.read_csv(countix_path)
    class_names = {'bench pressing','exercising arm','front raises','jumping jacks','lunge','mountain climber (exercise)','pull ups'
                   'push up','rope pushdown','situp','squat'}
    countix_df = countix[countix['class'].isin(class_names)]
    # idx = 3
    # url = 'https://youtu.be/{0}'.format(countix_df.iloc[idx]['video_id'])
    # start_time = countix_df.iloc[idx]['repetition_start']
    # end_time = countix_df.iloc[idx]['repetition_end']
    # download_video_from_url(idx,start_time=start_time, end_time=end_time, url_to_video=url)
    print(countix_df.shape[0])
    p = Pool(6)
    for i in range(countix_df.shape[0]):
        
        # url = 'https://www.youtube.com/watch?v={0}'.format(countix.iloc[i]['video_id'])
        start_time = countix_df.iloc[i]['repetition_start']
        end_time = countix_df.iloc[i]['repetition_end']
        url = 'https://youtu.be/{0}'.format(countix_df.iloc[i]['video_id'])
        p.apply_async(download_video_from_url, args=(i,start_time,end_time,url,video_dir))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    if os.path.exists(csv_path):
        os.remove(csv_path)
    countix_df.to_csv(csv_path,index=False)
    print('All subprocesses done.')
