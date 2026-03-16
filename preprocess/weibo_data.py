import os
import json
import re
from tqdm import tqdm

def remove_urls_from_content(content):
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', content).strip()

def remove_mention_from_content(content):
    mention_pattern = r'@\S+ '
    return re.sub(mention_pattern, '', content).strip()

# img_text中用户名
def remove_mention_from_imgtext(content):
    mention_pattern = r'@[^ \n]+[ \n]?'
    return re.sub(mention_pattern, '', content).strip()

def remove_hashtag_from_content(content):
    hashtag_pattern = r'#.*?#'
    special_pattern = r'【#.*?#】'
    content = re.sub(special_pattern, '', content)
    return re.sub(hashtag_pattern, '', content).strip()

def filter_json_and_images(folder_path):
    global all_len, img_count, max_len, max_len_img, min_len, min_len_img
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    
    for json_file in tqdm(json_files):
        json_path = os.path.join(folder_path, json_file)

        # 加载 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            post_info = json.load(f)

        # 检查 post_content 和 recognized_text 的长度和内容
        # post_content = post_info.get("post_content", "")
        post_content = post_info.get("recognized_text", "")
        post_content_no_mention = remove_mention_from_imgtext(post_content)
        post_clean_content = remove_urls_from_content(post_content_no_mention)
        post_clean_content = remove_hashtag_from_content(post_clean_content) # 微博
        
        if len(post_clean_content) > max_len:
            max_len = len(post_clean_content)
        if len(post_clean_content) < min_len:
            min_len = len(post_clean_content)
        if len(post_clean_content) > 30:
            max_len_img += 1
        if len(post_clean_content) < 6:
            min_len_img += 1
            print(len(post_clean_content),post_clean_content)
        all_len = all_len + len(post_clean_content)
        img_count = img_count + 1


data_path = "weibo_data/" # 存放所有weibo图片和对应的json文件

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
print("img_count",img_count) # post:筛除<20和>500,img_text:筛除<6和>30
print("max_len_img:",max_len_img)
print("min_len_img:",min_len_img)
print("average_len:",all_len/img_count)