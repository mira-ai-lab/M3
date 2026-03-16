import os
import json
import re
from tqdm import tqdm

def remove_urls_from_content(content):
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', content).strip()

def remove_mention_from_content(content):
    mention_pattern = r'@\S+'
    return re.sub(mention_pattern, '', content).strip()

def remove_hashtag_from_content(content):
    hashtag_pattern = r'#\S+'
    return re.sub(hashtag_pattern, '', content).strip()

def replace_newlines(content):
    content = re.sub(r'\n\n',',',content)
    content = re.sub(r'\n',',',content)
    return content

def filter_json_and_images(folder_path):
    global all_len, img_count, max_len, max_len_img
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    
    for json_file in tqdm(json_files):
        json_path = os.path.join(folder_path, json_file)

        # 加载 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            post_info = json.load(f)

        # 检查 post_content 的长度和内容
        post_content = post_info.get("post_content", "")
        post_content = replace_newlines(post_content)
        post_content_no_mention = remove_mention_from_content(post_content)
        post_clean_content = remove_urls_from_content(post_content_no_mention)
        post_clean_content = remove_hashtag_from_content(post_clean_content)
        
        # print(len(post_clean_content), post_clean_content, json_file, post_info.get("post_img"))

        if len(post_clean_content) > max_len:
            max_len = len(post_clean_content)
        if len(post_clean_content) > 250:
            max_len_img += 1
        all_len = all_len + len(post_clean_content)
        img_count = img_count + 1


data_path = "twitter_data/" # 存放所有twitter图片和对应的json文件

all_len = 0
img_count = 0
max_len = 0
max_len_img = 0

filter_json_and_images(data_path)

print(max_len)
print(all_len)
print(img_count) # 28187
print(max_len_img) # >250的1704张
print(all_len/img_count)