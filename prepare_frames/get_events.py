import cv2
import pandas as pd
import os

# def runBash(command):
# 	os.system(command)

# def crop(start,end,input,output):
# 	str = "ffmpeg -i " + input + " -ss  " + start + " -to " + end + " -c copy " + output
# 	print str
# 	runBash(str)


games = ['2019-01-03_MIN_at_TOR', '2019-01-07 NSH at TOR', 
	'2019-02-17_phi_at_det', '2019-02-25_mtl_at_njd']
game_ids = ['2018020621', '2018020652', '2018020907', '2018020963']
csv_files = ['2019-01-03 min@tor.csv', '2019-01-07 nsh@tor.csv',
	'Jul25 2019 New Game Files (15 games).csv', 'Jul25 2019 New Game Files (15 games).csv']

for i in range(len(games)):
	if i == 0:
		continue

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

	for index, row in df.iterrows():
		if str(row['nhl_game_id']) == game_ids[i]:
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

					input_video = vid_file.format(games[i], period)
					output_video = output_vid_file.format(game, period, str(vid_count).zfill(4))
					output_audio = output_audio_file.format(game, period, str(vid_count).zfill(4))

					run_str = 'ffmpeg -i "{}" -ss {} -to {} -c copy {}'.format(input_video, start_time, end_time, output_video)

					os.system(run_str)

					for r in rows:
						r['clip_number'] = vid_count
						output_df = output_df.append(r)

					rows = []

					# ffmpeg -i input.mkv -vn audio_only.ogg
					run_str = 'ffmpeg -i "{}" -vn {}'.format(output_video, output_audio)
					os.system(run_str)

					vid_count += 1
					count = -1
				else:
					count = -1

			count += 1

	output_df.to_csv(output_csv)



exit()

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
