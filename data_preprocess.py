import argparse

from tensorflow.python.eager.context import num_gpus
from  Datasets.SyntheticDataset import synthetic_data_process_mobile, SyntheticDataGenerator
from Datasets.ContixDataset import combined_data_preprocess
from Datasets.ContixDataset import view_test_vid as view_contix_vid
from Datasets.SyntheticDataset import view_test_vid as view_synthetix_vid
# synthetic_data_process_mobile('G:/RepNet/data/synthvids/','G:/RepNet/RepNetData10/',sample_size=10, start_idx=0)

# generator = SyntheticDataGenerator('G:/RepNet/data/synthvids/','G:/RepNet/RepNetData10/',sample_size=100, start_idx=0)
# generator.generate()
# combined_data_preprocess('G:/RepNet/data/trainvids', 'G:/RepNet/data/trainvids/countix_df.csv',save_path='G:/RepNet/RepNetData9/',sample_size=100,start_idx = 0)
# view_contix_vid('G:/RepNet/data/trainvids','G:/RepNet/data/trainvids/countix_df.csv','C:/Users/11351/Desktop/test.avi')
# view_synthetix_vid('G:/RepNet/data/synthvids/','C:/Users/11351/Desktop/test.avi')

if __name__ == '__main__':
    generator = SyntheticDataGenerator('G:/RepNet/data/synthvids/','G:/RepNet/RepNetData10/',sample_size=10, start_idx=0, cpu_num=2)
    generator.generate()