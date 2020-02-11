import pandas as pd
import pickle

dataset_dir = '/home/pascale/Documents/courses/CS886/MTL-AQA/MTL-AQA_dataset_release/Ready_2_Use/smaller_training_sets/size_140/'
train_split = pd.read_pickle(dataset_dir + 'train_split_0.pkl')


output_csv = '/home/pascale/Documents/courses/CS886/output_clips/2019-01-03_MIN_at_TOR.csv'
hockey_annotations = pd.read_csv(output_csv, index_col = 0)

last_clip = int(list(hockey_annotations['clip_number'])[-1])

train = range(last_clip)

with open('/home/pascale/Documents/courses/CS886/output_clips/train_data/train_split.pkl', 'wb') as f:
	pickle.dump(train, f)
