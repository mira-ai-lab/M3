import os
import sys
import pandas as pd
import json
from tqdm import tqdm
import shutil

sys.stdout.reconfigure(encoding='utf-8')

def rename_and_move_img(old_path, new_path):
    if not os.path.exists(new_path):
        shutil.copy(old_path, new_path)

def write_post(post_info, save_path, ix):
    file_path = os.path.join(save_path, post_info['user_id'] + "_" + str(ix) + '.json')
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(post_info, f, indent=4, ensure_ascii=False)

def pack_post(id, ix, row, url, img_path, save_path):
    new_img_name = str(id) + '_' + str(ix) + ".jpg"
    post_info = {
        'user_id': id,
        'post_id': row.iloc[0],
        'post_content': row.iloc[1],
        'post_img': new_img_name,
        'post_time': row.iloc[6],
        'img_url': url
    }
    write_post(post_info, save_path, ix)
    rename_and_move_img(os.path.join(img_path, url.split("/")[-1]),
                      os.path.join(save_path, new_img_name))

def get_second_level_folders(base_folder):
    second_level_folders = []
    for sub_dir in os.listdir(base_folder):
        sub_dir_path = os.path.join(base_folder, sub_dir)
        if os.path.isdir(sub_dir_path):
            for item in os.listdir(sub_dir_path):
                item_path = os.path.join(sub_dir_path, item)
                if os.path.isdir(item_path):
                    relative_path = os.path.relpath(item_path, base_folder)
                    second_level_folders.append(relative_path)
    return second_level_folders

base_folder = 'weibo/' # 存放原始weibo数据的文件夹

# 获取所有二级文件夹
second_level_folders = get_second_level_folders(base_folder)

print(f"Second level folders:{len(second_level_folders)}")
print(second_level_folders)
print([i.split('/')[1]+'.csv' for i in second_level_folders])
save_path = 'weibo_data/' # 存放所有weibo图片和对应的json文件
for i in tqdm(second_level_folders):
    id = i.split('/')[1]
    csv_path = os.path.join(base_folder,i,i.split('/')[1]+'.csv')
    data = pd.read_csv(csv_path)
    ex = os.listdir(os.path.join(base_folder,i,'img'))
    if ex:
        img_path = os.path.join(base_folder,i,'img/',ex[0])
    else:
        img_path = ""
    # 遍历DataFrame的每一行，获取第二列内容
    ix = 0
    for _, row in data.iterrows():
        img_urls = row.iloc[3]
        post_content = row.iloc[1]
        if type(img_urls) is float:
            continue
        if "无" not in img_urls and post_content is not None:
            if "," in img_urls:
                img_urls = img_urls.split(",")
                for i, url in enumerate(img_urls):
                    if i >= 2:
                        break
                    check_img = os.path.join(img_path, url.split("/")[-1])
                    if os.path.exists(check_img):
                        pack_post(id, ix, row, url, img_path, save_path)
                        ix += 1

            else:
                check_img = os.path.join(img_path, img_urls.split("/")[-1])
                if os.path.exists(check_img):
                    pack_post(id, ix, row, img_urls, img_path, save_path)
                    ix += 1
    #print(csv_path)