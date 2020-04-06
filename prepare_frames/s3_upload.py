import logging
import boto3
from botocore.exceptions import ClientError
import os


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

audio_dir = '/home/pascale/Documents/courses/CS886/final_project/output_clips/audio'
bucket = 'pascale-hockey'

files = os.listdir(audio_dir)

for f in files:
    if '2019-01-03_MIN_at_TOR' in f:
        continue
    elif '2019-01-07_NSH_at_TOR' in f:
        upload_file(os.path.join(audio_dir, f), bucket, os.path.join('2019-01-07_NSH_at_TOR', f))
    elif '2019-02-17_phi_at_det' in f:
        upload_file(os.path.join(audio_dir, f), bucket, os.path.join('2019-02-17_phi_at_det', f))
    elif '2019-02-25_mtl_at_njd' in f:
        upload_file(os.path.join(audio_dir, f), bucket, os.path.join('2019-02-25_mtl_at_njd', f))
