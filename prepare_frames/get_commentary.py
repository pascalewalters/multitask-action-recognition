import pandas as pd
import os
import json
import pickle


dataset_dir = '/home/pascale/Documents/courses/CS886/MTL-AQA/MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/'
final_captions_dict = pd.read_pickle(dataset_dir + 'final_captions_dict.pkl')

print(final_captions_dict[(1, 1)])

transcript_dir = '/home/pascale/Documents/courses/CS886/output_clips/transcribe/'

files = os.listdir(transcript_dir)
files.sort()

captions_dict = {}
count = 0

for f in files:
	with open(os.path.join(transcript_dir, f), 'r') as json_file:
		caption = ['<sos>']

		data = json.load(json_file)
		assert len(data['results']['transcripts']) == 1

		transcript = data['results']['transcripts'][0]['transcript']
		transcript = transcript.replace('.', '')
		transcript = transcript.replace(',', '')
		transcript = transcript.lower()

		caption.extend(transcript.split())
		caption.append('<eos>')

		captions_dict[count] = caption
		count += 1


with open('/home/pascale/Documents/courses/CS886/output_clips/train_data/captions_dict.pkl', 'wb') as f:
	pickle.dump(captions_dict, f)

