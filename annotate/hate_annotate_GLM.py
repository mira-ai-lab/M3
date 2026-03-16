import base64
from zhipuai import ZhipuAI
import json
from tqdm import tqdm
import os
import re
import gc

client = ZhipuAI(api_key="") # 填写您自己的APIKey

prompt_template = """ You are a hateful meme detection assistant.
    Based on the input image and post: {post_text}, please select the most appropriate option from ["hate", "normal"].
    And give your confidence score for this option as a decimal between 0 and 1.
    Please output in this format:{"label": "hate" or "normal","confidence_score": 0.0-1.0}
"""

input_json_path = "preprocess/weibo_img_url.json"
image_folder = "weibo_data" # 存放所有weibo图片和对应的json文件
output_json_path = "weibo_hate_results_GLM4vflash.json"

CHUNK_SIZE = 200

# get json from text
def extract_json(text):
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        return None

# load existing results
if os.path.exists(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    processed_ids = {item["img"] for item in results}
else:
    processed_ids = set()
    results = []

with open(input_json_path, "r", encoding="utf-8") as f:
    image_data_list = json.load(f)

for i in tqdm(range(0, len(image_data_list), CHUNK_SIZE), desc="Processing in chunks"):
    chunk = image_data_list[i:i+CHUNK_SIZE]
    chunk_results = []

    for image_data in chunk:
        image_id = image_data["img"]
        if image_id in processed_ids:
            continue

        try:
            image_path = os.path.join(image_folder, image_id)
            with open(image_path, 'rb') as img_file:
                img_base = base64.b64encode(img_file.read()).decode('utf-8')
            json_path = os.path.join(image_folder, image_id.replace('.jpg', '.json'))
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    post_data = json.load(f)
                post_text = post_data.get("post_content", "")
            else:
                post_text = ""
        
            response = client.chat.completions.create(
                model="glm-4v-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_base}
                        },
                        {
                            "type": "text",
                            "text": prompt_template
                        }
                        ]
                    }
                ]
            )

            response_text = response.choices[0].message.content
            result = extract_json(response_text)
            if result is None:
                result = {"label": "uncertain","confidence_score": 0.0}

            chunk_results.append({
                "img": image_id,
                "label": result.get("label", "uncertain"),
                "confidence_score": result.get("confidence_score", 0.0),
                "raw_output": response_text
            })

        except Exception as e:
            chunk_results.append({
                "img": image_id,
                "predicted_category": "error",
                "confidence_score": 0.0,
                "raw_output": "None",
                "error": str(e)
            })

        processed_ids.add(image_id)

    results.extend(chunk_results)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    # release memory
    del chunk, chunk_results
    gc.collect()