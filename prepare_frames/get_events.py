import cv2
import pandas as pd
import os

# def runBash(command):
# 	os.system(command)

# def crop(start,end,input,output):
# 	str = "ffmpeg -i " + input + " -ss  " + start + " -to " + end + " -c copy " + output
# 	print str
# 	runBash(str)


vid_file = '/home/pascale/Documents/courses/CS886/2019-01-03_MIN_at_TOR_P{}.MP4'
output_vid_file = '/home/pascale/Documents/courses/CS886/output_clips/2019-01-03_MIN_at_TOR_P{}_{}.MP4'
output_audio_file = '/home/pascale/Documents/courses/CS886/output_clips/audio/2019-01-03_MIN_at_TOR_P{}_{}.mp4'

# Read in CSV
df = pd.read_csv('/home/pascale/Documents/courses/CS886/event_data/2019-01-03 min@tor.csv')
output_csv = '/home/pascale/Documents/courses/CS886/output_clips/2019-01-03_MIN_at_TOR.csv'
output_df = pd.DataFrame()

count = 0
vid_count = 0

rows = []

for index, row in df.iterrows():

	period = row['period']
	time = row['video_clock_seconds']

	m, s = divmod(time, 60)

	rows.append(row)

	if count == 0:
		start_time = '00:{}:{}'.format(str(m).zfill(2), str(s).zfill(2))
		period_1 = period
	elif count == 9:
		if period_1 == period:
			end_time = '00:{}:{}'.format(str(m).zfill(2), str(s).zfill(2))

			input_video = vid_file.format(period)
			output_video = output_vid_file.format(period, str(vid_count).zfill(4))
			output_audio = output_audio_file.format(period, str(vid_count).zfill(4))

			# run_str = 'ffmpeg -i {} -ss {} -to {} -c copy {}'.format(input_video, start_time, end_time, output_video)
			# os.system(run_str)

			for r in rows:
				r['clip_number'] = vid_count
				output_df = output_df.append(r)

			rows = []

			# ffmpeg -i input.mkv -vn audio_only.ogg
			run_str = 'ffmpeg -i {} -vn {}'.format(output_video, output_audio)
			# os.system(run_str)

			vid_count += 1
			count = -1
		else:
			count = -1

	count += 1

output_df.to_csv(output_csv)
