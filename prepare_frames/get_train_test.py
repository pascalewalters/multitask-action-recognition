import pandas as pd
import pickle
import random


output_csv = '/home/pascale/Documents/courses/CS886/final_project/output_clips/2019-01-03_MIN_at_TOR.csv'
hockey_annotations = pd.read_csv(output_csv, index_col = 0)

last_clip = int(list(hockey_annotations['clip_number'])[-1])

indices = list(range(last_clip))
random.shuffle(indices)
split_idx = int(last_clip * 0.7)

train = indices[:split_idx]
test = indices[split_idx:]

print('Train:', len(train))
print('Test:', len(test))

with open('/home/pascale/Documents/courses/CS886/final_project/output_clips/train_data/train_split_0.pkl', 'wb') as f:
	pickle.dump(train, f)

with open('/home/pascale/Documents/courses/CS886/final_project/output_clips/train_data/test_split_0.pkl', 'wb') as f:
	pickle.dump(test, f)
