# declaring random seed
randomseed = 0

# directory containing dataset annotation files; this anno_n_splits_dir make the full path
# dataset_dir = '/home/pascale/Documents/courses/CS886/MTL-AQA/MTL-AQA_dataset_release/Ready_2_Use/'
dataset_dir = '/home/pascale/Documents/courses/CS886/final_project/output_clips/'

# directory tp store train/test split lists and annotations
# anno_n_splits_dir = dataset_dir + 'smaller_training_sets/size_140/'
anno_n_splits_dir = dataset_dir + 'train_data'

# directory containing extracted frames
# dataset_frames_dir = anno_n_splits_dir + 'frames/'
dataset_frames_dir = dataset_dir + 'frames/'

# directory for saving output
saving_dir = 'output8/'

# sample length in terms of no of frames
# sample_length = 103
sample_length = 25

# input data dims; C3D-AVG:112; MSCADC: 180
C, H, W = 3,112,112
# C, H, W = 3,180,180#3,112,112#
# image resizing dims; C3D-AVG: 171,128; MSCADC: 640,360
input_resize = 171, 128
# input_resize = 640,360#171,128#
# temporal augmentation range
temporal_aug_min = -3; temporal_aug_max = 3

# score std
final_score_std = 17

# maximum caption length
max_cap_len = 100

# vocab_size = 5779
vocab_size = 2197

caption_lstm_dim_hidden = 512
caption_lstm_dim_word = 512
caption_lstm_dim_vid = 8192 # C3D-AVG: 8192; MSCADC: 1200
# caption_lstm_dim_vid = 1200#8192# C3D-AVG: 8192; MSCADC: 1200
caption_lstm_cell_type = 'gru'
# caption_lstm_cell_type = 'lstm'
caption_lstm_num_layers = 2
caption_lstm_dropout = 0.5
caption_lstm_lr = 0.0001

# task 2 include
with_dive_classification = False
with_hockey_classification = True
with_caption = True
with_score_regression = False

max_epochs = 100

train_batch_size = 3
test_batch_size = 5

model_ckpt_interval = 10 # in epochs

base_learning_rate = 0.0001

temporal_stride = 16

class_occurences = {
	'PlayMakeEvent':       864,
	'PlayReceiveEvent':    864,
	'SwitchEvent':         584,
	'ShotEvent':           144,
	'AdvanceEvent':        128,
	'FaceoffEvent':         79,
	# 'HitEvent':             79,
	# 'WhistleEvent':         68,
	# 'ShotBlockEvent':       42,
	# 'PenaltyEvent':          7,
	# 'RicochetEvent':         1
}

