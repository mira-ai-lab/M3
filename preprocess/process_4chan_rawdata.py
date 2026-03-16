import json
import os
import shutil
import json
from PIL import Image
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def convert_image_to_jpg(input_path, output_path):
    with Image.open(input_path) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(output_path, format='JPEG')


def convert_webm_to_jpg(input_path, output_path, frame_time=0):
    clip = VideoFileClip(input_path)
    frame = clip.get_frame(frame_time)
    frame_img = Image.fromarray(frame)
    frame_img.save(output_path, format='JPEG')


def rename_and_move_img(old_path, new_path):
    if not os.path.exists(new_path):
        if '.webm' in old_path:
            convert_webm_to_jpg(old_path, new_path)
        else:
            convert_image_to_jpg(old_path, new_path)


def write_post(post_info, save_path, timestamp):
    file_path = os.path.join(save_path, post_info['thread_id'] + "_" + str(timestamp) + '.json')
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(post_info, f, indent=4, ensure_ascii=False)


def pack_post(json_data, img_path, save_path):
    id = json_data["thread_num"]
    content = json_data["comment"]["text"]
    title = json_data["comment"]["title"]
    timestamp = json_data["timestamp"]
    url = json_data["media"]["url"]

    new_img_name = str(id) + '_' + str(timestamp) + ".jpg"
    post_info = {
        'thread_id': id,
        'post_id': id,
        'post_title': title,
        'post_content': content,
        'post_img': new_img_name,
        'post_time': timestamp,
        'img_url': url
    }
    write_post(post_info, save_path, timestamp)
    rename_and_move_img(os.path.join(img_path, url.split("/")[-1]),
                        os.path.join(save_path, new_img_name))


base_folder = '4chan/' # 存放原始4chan数据的文件夹
subfolders = ["4chan24012402/4chan", "4chan2308"] # 原始4chan数据的子文件夹列表
save_path = '4chan_data/' # 存放所有4chan图片和对应的json文件
file_list = []

for i in subfolders:
    data_path = os.path.join(base_folder, i)
    data_json = os.path.join(data_path, "json")
    all_json_normal = [os.path.join(data_json,i) for i in os.listdir(data_json)]
    file_list.extend(all_json_normal)


for i in tqdm(file_list):
    #print(i)
    with open(i, 'r', encoding='utf-8') as file:
        data = json.load(file)
    image_folder = os.path.join("/".join(i.split("/")[:-2]),"media")
    for j in data:
        if j.get("media") is not None:
            if j["media"].get("url"):
                img_url = j["media"].get("url")
                id = j["thread_num"]
                timestamp = j["timestamp"]
                img_info = str(id) + '_' + str(timestamp)
                image_name = j["media"].get("url").split("/")[-1]
                check_img = os.path.join(image_folder, image_name)
                if os.path.exists(check_img):
                     pack_post(j,image_folder,save_path)