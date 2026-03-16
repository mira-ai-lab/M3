import os
import json
from tqdm import tqdm

def img2url(folder_path, output_file):
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    img_url_data = []
    
    for json_file in tqdm(json_files):
        json_path = os.path.join(folder_path, json_file)

        # 读取 JSON 文件
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                post_info = json.load(f)
                
            # 获取图片和 URL信息
            img = post_info.get("post_img")
            url = post_info.get("img_url", "")
        
            if img and url:
                img_url_data.append({"img": img, "url": url})
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # 重新写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(img_url_data, f, ensure_ascii=False)


data_path = "4chan_data/" # 存放所有4chan图片和对应的json文件
output_json = "4chan_img_url_data.json"
img2url(data_path, output_json)
