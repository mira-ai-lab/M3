import os
import json

# json_file_path = 'OCR/fail_4chan.json'
# images_folder_path = '4chan_data'
# json_file_path = 'OCR/fail_weibo.json'
# images_folder_path = 'weibo_data'
json_file_path = 'OCR/fail_twitter.json'
images_folder_path = 'twitter_data'

# Load the JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Iterate over the filenames in the JSON data
for item in data:
    filename = item.get('filename')
    jsonname = filename.replace('.jpg', '.json')
    if filename:
        img_path = os.path.join(images_folder_path, filename)
        json_path = os.path.join(images_folder_path, jsonname)
        # Check if the file exists and delete it
        if os.path.exists(img_path):
            os.remove(img_path)
            os.remove(json_path)
            print(f"Deleted: {img_path} and {json_path}")
        else:
            print(f"File not found: {img_path} and {json_path}")