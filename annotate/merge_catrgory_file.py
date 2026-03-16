import json

# 读取 URL 数据 
with open('preprocess/4chan_hate_img2url.json', 'r') as f: # twitter, weibo
    url_data = json.load(f)
    img_to_url = {item['img']: item['url'] for item in url_data}

merged_data = {}

# 合并category结果
model_result_paths = [
    'annotate/4chan_category_results_GLM4vflash.json',
    'annotate/4chan_category_results_llava.json',
    'annotate/4chan_category_results_qwen.json',
]

for path in model_result_paths:
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        for item in results:
            img = item['img']

            if img not in merged_data:
                merged_data[img] = {
                    'img': img,
                    'url': img_to_url.get(img, ""),
                    'category': [],
                    'confidence_score': []
                }

            if 'error' in item or item.get('predicted_category') == 'error' or not item.get('category'):
                merged_data[img]['category'].append('none')
                merged_data[img]['confidence_score'].append(0.0)
            else:
                merged_data[img]['category'].append(item['category'])
                merged_data[img]['confidence_score'].append(item['confidence_score'])

final_output = list(merged_data.values())

with open('annotate/4chan_category_results.json', 'w') as f:
    json.dump(final_output, f, ensure_ascii=False)