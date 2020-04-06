import cv2
import pandas as pd
import os
import datetime


games = ['2019-01-03_MIN_at_TOR']
game_ids = ['2018020621']
csv_files = ['2019-01-03 min@tor.csv']

for i in range(len(games)):
	game = games[i].replace(' ', '_')

	vid_file = '/home/pascale/Documents/courses/CS886/final_project/videos/{} P{}.MP4'
	output_vid_file = '/home/pascale/Documents/courses/CS886/final_project/output_clips/{}_P{}_{}.MP4'
	output_audio_file = '/home/pascale/Documents/courses/CS886/final_project/output_clips/audio/{}_P{}_{}.mp4'

	# Read in CSV
	df = pd.read_csv('/home/pascale/Documents/courses/CS886/final_project/event_data/{}'.format(csv_files[i]))
	output_csv = '/home/pascale/Documents/courses/CS886/final_project/output_clips/{}.csv'.format(game)
	output_df = pd.DataFrame()

	count = 0
	vid_count = 0

	rows = []
	clip_lengths = []
	events = []

	for index, row in df.iterrows():
		if str(row['nhl_game_id']) == game_ids[i]:
			period = row['period']
			time = row['video_clock_seconds']

			m, s = divmod(time, 60)
			rows.append(row)

			if count == 0:
				# start_time = '00:{}:{}'.format(str(m).zfill(2), str(s).zfill(2))
				start_time = datetime.time(hour = 0, minute = m, second = s)
				start_time = datetime.datetime.combine(datetime.date.today(), start_time)
				period_1 = period
			elif count == 9:
				if period_1 == period:
					# end_time = '00:{}:{}'.format(str(m).zfill(2), str(s).zfill(2))
					end_time = datetime.time(hour = 0, minute = m, second = s)
					end_time = datetime.datetime.combine(datetime.date.today(), end_time)

					clip_length = end_time - start_time
					clip_lengths.append(clip_length.total_seconds())

					# input_video = vid_file.format(games[i], period)
					# output_video = output_vid_file.format(game, period, str(vid_count).zfill(4))
					# output_audio = output_audio_file.format(game, period, str(vid_count).zfill(4))

					# run_str = 'ffmpeg -i "{}" -ss {} -to {} -c copy {}'.format(input_video, start_time, end_time, output_video)
					# os.system(run_str)

					for r in rows:
						r['clip_number'] = vid_count
						output_df = output_df.append(r)

					rows = []

					# ffmpeg -i input.mkv -vn audio_only.ogg
					# run_str = 'ffmpeg -i "{}" -vn {}'.format(output_video, output_audio)
					# os.system(run_str)

					vid_count += 1
					count = -1
				else:
					count = -1

			count += 1

	# output_df.to_csv(output_csv)
	print('Unique events:', output_df['eventable_type'].unique())
	print(output_df['eventable_type'].value_counts())

	print('Number of clips:', len(clip_lengths))
	print('Average clip length:', sum(clip_lengths) / len(clip_lengths))
	print('Maximum clip length:', max(clip_lengths))
	print('Minimum clip length:', min(clip_lengths))
