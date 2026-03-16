import os
import json
from tqdm import tqdm

def filter_meme(folder_path):
    count = 0
    notmemes = []
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    print("total:",len(json_files))

    for json_file in tqdm(json_files):
        json_path = os.path.join(folder_path, json_file)
        img_file = json_file.replace('.json', '.jpg')
        img_path = os.path.join(folder_path, img_file)

        if os.path.exists(img_path) == False:
            print(f"File not found: {img_path}")
            # os.remove(json_path)
            continue

        # 加载 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            post_info = json.load(f)

        # 检查recognized_text内容
        try:
            recognized_text = post_info["recognized_text"]
        except KeyError:
            recognized_text = None

        if not recognized_text or recognized_text == "None":
            count += 1
            notmemes.append({"img": img_file, "json": json_file})
            os.remove(img_path)
            os.remove(json_path)
            # print(f"Delete file: {img_path} and {json_path}")

        # only weibo_user_id in recognized_text
        # if recognized_text.startswith("@") and len(recognized_text) < 30: # 微博用户名不超过30字符
        #     notmemes.append({"img": img_file, "json": json_file})
        #     os.remove(img_path)
        #     os.remove(json_path)
    
    with open("notMeme_weibo.json", 'w', encoding='utf-8') as f:
        json.dump(notmemes, f, ensure_ascii=False, indent=4)
    print("count:", count)


# data_path = "twitter_data/" # 存放所有twitter图片和对应的json文件
data_path = "weibo_data/" # 存放所有weibo图片和对应的json文件
# data_path = "4chan_data/" # 存放所有4chan图片和对应的json文件
filter_meme(data_path)

