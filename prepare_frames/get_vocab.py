import json
import pandas as pd


dataset_dir = '/home/pascale/Documents/courses/CS886/MTL-AQA/MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/'
final_captions_dict = json.load(open(dataset_dir + 'vocab.json'))

# print(len(final_captions_dict['ix_to_word'].keys()))

captions_dict = pd.read_pickle('/home/pascale/Documents/courses/CS886/output_clips/train_data/captions_dict.pkl')

vocab = set()

for v in captions_dict.values():
	vocab.update(v)

vocab.remove('<eos>')
vocab.remove('<sos>')

ix_to_word = {}
word_to_ix = {}

for i, word in enumerate(vocab):
	ix_to_word[i] = word
	word_to_ix[word] = i

vocab_dict = {'ix_to_word': ix_to_word,
			  'word_to_ix': word_to_ix}

json.dump(vocab_dict, open('/home/pascale/Documents/courses/CS886/output_clips/train_data/vocab.json' ,'w'))
