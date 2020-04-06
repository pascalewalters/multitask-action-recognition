import boto3
import os
import time
import urllib.request


audio_dir = '/home/pascale/Documents/courses/CS886/final_project/output_clips/audio'
transcript_dir = '/home/pascale/Documents/courses/CS886/final_project/output_clips/transcribe'

files = os.listdir(audio_dir)
files.sort()

transcribe = boto3.client('transcribe')

idx = 0

while idx <= len(files):

	jobs = transcribe.list_transcription_jobs(MaxResults = 100, Status = 'IN_PROGRESS')

	if len(jobs['TranscriptionJobSummaries']) < 100:
		# Maximum 100 jobs	
		f = files[idx]

		try:
			# Download transcribed file
			status = transcribe.get_transcription_job(TranscriptionJobName = f)

			url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
			
			urllib.request.urlretrieve(url, os.path.join(transcript_dir, f.replace('mp4', 'json')))

		except Exception as e:
			# Start next job
			if '2019-01-03_MIN_at_TOR' in f:
				continue
			elif '2019-01-07_NSH_at_TOR' in f:
				job_uri = 's3://pascale-hockey/2019-01-07_NSH_at_TOR/' + f
				vocab_name = '2019-01-07_NSH_at_TOR'
				# print('here')
			elif '2019-02-17_phi_at_det' in f:
				continue
				# job_uri = 's3://pascale-hockey/2019-02-17_phi_at_det/' + f
				# vocab_name = '2019-02-17_phi_at_det'
			elif '2019-02-25_mtl_at_njd' in f:
				continue
				# job_uri = 's3://pascale-hockey/2019-02-25_mtl_at_njd/' + f
				# vocab_name = '2019-02-25_mtl_at_njd'

			transcribe.start_transcription_job(
				TranscriptionJobName = f,
				Media = {'MediaFileUri': job_uri},
				MediaFormat = 'mp4',
				LanguageCode = 'en-US',
				Settings = {'VocabularyName': vocab_name}
			)

		idx += 1

	time.sleep(1)
