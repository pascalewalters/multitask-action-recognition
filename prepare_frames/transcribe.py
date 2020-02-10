import boto3
import os
import time
import urllib.request


audio_dir = '/home/pascale/Documents/courses/CS886/output_clips/audio'
transcript_dir = '/home/pascale/Documents/courses/CS886/output_clips/transcribe'

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
			job_uri = 's3://pascale-hockey/2019-01-03_MIN_at_TOR_P3/' + f

			transcribe.start_transcription_job(
				TranscriptionJobName = f,
				Media = {'MediaFileUri': job_uri},
				MediaFormat = 'mp4',
				LanguageCode = 'en-US'
			)

			time.sleep(1)

		idx += 1
