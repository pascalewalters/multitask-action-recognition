import cv2
import pandas as pd
import os


clips_dir = '/home/pascale/Documents/courses/CS886/output_clips'
files = os.listdir(clips_dir)
files.sort()

frames_dir = '/home/pascale/Documents/courses/CS886/output_clips/frames'

game = '2019-01-03_MIN_at_TOR_P3'
vid_count = 0

for f in files:
	if not f.endswith('MP4'):
		continue

	vidcap = cv2.VideoCapture(os.path.join(clips_dir, f))
	success, image = vidcap.read()
	count = 0

	while success:
		cv2.imwrite(os.path.join(frames_dir, 
			'{}_{}_{}.jpg'.format(game, str(vid_count).zfill(4), str(count).zfill(4))),
			image)

		success, image = vidcap.read()
		count += 1

	vidcap.release()
	vid_count += 1