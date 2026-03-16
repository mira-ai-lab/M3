import os
import json
import re
from tqdm import tqdm

def remove_urls_from_content(content):
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', content).strip()

def remove_reference_from_content(content):
    content = content.replace('>', '')
    reference_pattern = r'>>\d+\.'
    return re.sub(reference_pattern, '', content).strip()

def filter_json_and_images(folder_path):
    global all_len, img_count, max_len, max_len_img, min_len, min_len_img
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    
    for json_file in tqdm(json_files):
        json_path = os.path.join(folder_path, json_file)

        # 加载 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            post_info = json.load(f)

        # 检查 post_content 和recognized_text的长度和内容
        # post_content = post_info.get("post_content", "")
        post_content = post_info.get("recognized_text", "")
        post_content_no_mention = remove_reference_from_content(post_content)
        post_clean_content = remove_urls_from_content(post_content_no_mention)
        
        if len(post_clean_content) > max_len:
            max_len = len(post_clean_content)
        if len(post_clean_content) < min_len:
            min_len = len(post_clean_content)
        if len(post_clean_content) > 100: 
            max_len_img += 1
        if len(post_clean_content) < 10: 
            min_len_img += 1
        all_len = all_len + len(post_clean_content)
        img_count = img_count + 1


data_path = "4chan_data/" # 存放所有4chan图片和对应的json文件

all_len = 0
img_count = 0
max_len = 0
min_len = 10000
max_len_img = 0
min_len_img = 0

filter_json_and_images(data_path)

print("max_len:",max_len)
print("min_len:",min_len)
print("all_len:",all_len)
print("img_count",img_count) # 筛除post <20和>500的, 筛除recognized_text <10和>100的
print("max_len_img:",max_len_img) 
print("min_len_img:",min_len_img)
print("average_len:",all_len/img_count)
