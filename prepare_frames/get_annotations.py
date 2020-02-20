import pandas as pd
import pickle


dataset_dir = '/home/pascale/Documents/courses/CS886/MTL-AQA/MTL-AQA_dataset_release/Ready_2_Use/smaller_training_sets/size_140/'
final_annotations_dict = pd.read_pickle(dataset_dir + 'final_annotations_dict.pkl')

output_csv = '/home/pascale/Documents/courses/CS886/output_clips/2019-01-03_MIN_at_TOR.csv'
hockey_annotations = pd.read_csv(output_csv, index_col = 0)

last_clip = int(list(hockey_annotations['clip_number'])[-1])

hockey_annotations_dict = {}

for i in range(last_clip):
	clip_events = hockey_annotations.loc[hockey_annotations['clip_number'] == float(i)]
	clip_events = list(clip_events['eventable_type'])
	
	annotations_dict = {'SwitchEvent': 0,
						'AdvanceEvent': 0,
						'FaceoffEvent': 0,
						'PlayMakeEvent': 0,
						'PlayReceiveEvent': 0,
						'WhistleEvent': 0,
						'ShotEvent': 0,
						'HitEvent': 0,
						'ShotBlockEvent': 0,
						'PenaltyEvent': 0,
						'RicochetEvent': 0
						}

	for event in clip_events:
		annotations_dict[event] += 1

	hockey_annotations_dict[i] = annotations_dict

with open('/home/pascale/Documents/courses/CS886/output_clips/train_data/final_annotations_dict.pkl', 'wb') as f:
	pickle.dump(hockey_annotations_dict, f)
