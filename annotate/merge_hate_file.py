import json

# 读取 URL 数据 
with open('preprocess/weibo_img_url_data.json', 'r', encoding='utf-8') as f: # twitter_img_url_data.json
    url_data = json.load(f)
    img_to_url = {item['img']: item['url'] for item in url_data}

merged_data = {}

# 合并hate结果
model_result_paths = [
    'annotate/weibo_hate_results_GLM4vflash.json',
    'annotate/weibo_hate_results_qwen.json',
    'annotate/weibo_hate_results_llava.json'
]
for path in model_result_paths:
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        for item in results:
            img = item['img']

            if 'error' in item or item.get('predicted_category') == 'error':
                label = 'hate'
                score = 1.0
            else:
                label = item['label']
                score = item['confidence_score']

            if img not in merged_data:
                merged_data[img] = {
                    'img': img,
                    'url': img_to_url.get(img, ""),
                    'label': [],
                    'confidence_score': []
                }
            merged_data[img]['label'].append(label)
            merged_data[img]['confidence_score'].append(score)

final_output = list(merged_data.values())

with open('annotate/weibo_hate_results.json', 'w', encoding='utf-8') as f:  # 合并后的文件
    json.dump(final_output, f, ensure_ascii=False)