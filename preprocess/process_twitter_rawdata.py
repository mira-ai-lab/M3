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

def pack_post(id, ix, row, saved_filename, img_path, save_path):
    new_img_name = str(id) + '_' + str(ix) + ".jpg"
    post_info = {
        'user_id': id,
        'post_id': row.iloc[1],  #display name
        'post_content': row.iloc[7],  #tweet content
        'post_img': new_img_name,
        'post_time': row.iloc[0],  #tweet date
        'img_url': row.iloc[5] # Media URL
    }
    write_post(post_info, save_path, ix)
    rename_and_move_img(os.path.join(img_path, saved_filename),
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

base_folder = 'twitter/' # 存放原始twitter数据的文件夹

# 获取所有二级文件夹
second_level_folders = get_second_level_folders(base_folder)

print(f"Second level folders:{len(second_level_folders)}")
print(second_level_folders)  
print([i.split('/')[1]+'.csv' for i in second_level_folders])
# print([os.listdir(os.path.join(base_folder,i)) for i in second_level_folders])
save_path = 'twitter_data/' # 存放所有twitter图片和对应的json文件
for i in tqdm(second_level_folders):
    id = i.split('/')[1]
    csv_path = os.path.join(base_folder,i,i.split('/')[1]+'.csv')
    # try:
    #     data = pd.read_csv(csv_path)
    # except pd.errors.ParserError as e:
    #     print(e)
    data = pd.read_csv(csv_path, skiprows=3)
    img_path = os.path.join(base_folder,i)
    #遍历DataFrame的每一行，获取第二列内容
    ix = 0
    for _, row in data.iterrows():
        saved_filename = row.iloc[6] # 图片的命名（.csv中第7列Saved Filename）
        img_urls = row.iloc[5]  # 图片的URL（.csv中第6列Media URL）
        post_content = row.iloc[7] # twitter正文内容（.csv中第8列Tweet Content）
        # if type(img_urls) is float:
        #     continue
        if img_urls is not None and post_content is not None:
            check_img = os.path.join(img_path, saved_filename)
            if os.path.exists(check_img):
                pack_post(id, ix, row, saved_filename, img_path, save_path)
                ix += 1
    #print(csv_path)