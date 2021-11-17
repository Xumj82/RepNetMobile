from __future__ import generators
import argparse
from utils.hparams import HParam
from tensorflow.python.eager.context import num_gpus
from  Datasets.SyntheticDataset import SyntheticDataGenerator
from Datasets.ContixDataset import combined_data_preprocess
from Datasets.ContixDataset import view_test_vid as view_contix_vid



# generator = SyntheticDataGenerator('G:/RepNet/data/synthvids/','G:/RepNet/RepNetData10/',sample_size=100, start_idx=0)
# generator.generate()
# combined_data_preprocess('G:/RepNet/data/trainvids', 'G:/RepNet/data/trainvids/countix_df.csv',save_path='G:/RepNet/RepNetData9/',sample_size=100,start_idx = 0)
# view_contix_vid('G:/RepNet/data/trainvids','G:/RepNet/data/trainvids/countix_df.csv','C:/Users/11351/Desktop/test.avi')
# view_synthetix_vid('G:/RepNet/data/synthvids/','C:/Users/11351/Desktop/test.avi')
if __name__ == '__main__':
    hp = HParam('config/default.yaml')
    # synthetic_data_process_mobile(hp.prepocess.train_vids,hp.data.train_dir,sample_size=hp.prepocess.train_vids_size, start_idx=0)
    # combined_data_preprocess(hp.prepocess.test_vids, hp.prepocess.test_labels,save_path=hp.data.test_dir,sample_size=hp.prepocess.test_vids_size,start_idx = 0)
    generator = SyntheticDataGenerator(hp.prepocess.train_vids,
                                        hp.prepocess.mask_images,
                                        hp.data.train_dir,
                                        sample_size=hp.prepocess.train_vids_size, 
                                        start_idx=46,
                                        cpu_num=hp.prepocess.cpu_num)
    generator.generate()
    # generator.view_test_vid('C:/Users/11351/Desktop/test.avi')
    #     generator.view_pickle_vid('ckpt/000855_synthetic_data.pickle','C:/Users/11351/Desktop/test.avi')
    # view_synthetix_vid('G:/RepNet/data/synthvids/','G:/RepNet/data/masks/','C:/Users/11351/Desktop/test.avi')
