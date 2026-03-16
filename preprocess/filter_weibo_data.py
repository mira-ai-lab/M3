import os
import json
import re
from tqdm import tqdm

def remove_urls_from_content(content):
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', content).strip()

# post中用户名
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
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    
    for json_file in tqdm(json_files):
        json_path = os.path.join(folder_path, json_file)

        # 加载 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            post_info = json.load(f)

        # 检查 post_content 的长度和内容
        post_content = post_info.get("post_content", "")
        post_content_no_mention = remove_mention_from_content(post_content)
        post_clean_content = remove_urls_from_content(post_content_no_mention)
        post_clean_content = remove_hashtag_from_content(post_clean_content) # 微博
        
        # 检查 recognized_text 的长度和内容
        recognized_text = post_info.get("recognized_text", "")
        recognized_text_no_mention = remove_mention_from_imgtext(recognized_text)
        recognized_text_clean_content = remove_urls_from_content(recognized_text_no_mention)
        recognized_text_clean_content = remove_hashtag_from_content(recognized_text_clean_content) # 微博

        # print(len(post_clean_content), post_content, post_clean_content, json_file, post_info.get("post_img"))
        # print(len(recognized_text_clean_content), recognized_text_clean_content, json_file)
        if len(post_clean_content) < 20 or len(post_clean_content) > 500 :
            # 删除 JSON 文件
            os.remove(json_path)

            # 删除对应的图片文件
            post_img = post_info.get("post_img")
            if post_img:
                img_path = os.path.join(folder_path, post_img)
                if os.path.exists(img_path):
                    os.remove(img_path)

        if len(recognized_text_clean_content) < 6 or len(recognized_text_clean_content) > 30 :
            # 删除 JSON 文件
            os.remove(json_path)

            # 删除对应的图片文件
            post_img = post_info.get("post_img")
            if post_img:
                img_path = os.path.join(folder_path, post_img)
                if os.path.exists(img_path):
                    os.remove(img_path)

data_path = "weibo_data/" # 存放所有weibo图片和对应的json文件

filter_json_and_images(data_path)
