from io import BytesIO
import torch
import requests
from PIL import Image

from vllm import LLM, SamplingParams

import json
import os
import re
from tqdm import tqdm

# get json from text
def extract_json(text):
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        return None

# simple chunking function
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

if __name__ == "__main__":
    llm = LLM(model="LLaVA/llava-hf/llava-v1.6-mistral-7b-hf", max_model_len=4096, gpu_memory_utilization=0.8)

    input_json_path = "preprocess/4chan_hate_img2url.json" # twitter
    image_floder = "4chan_data" # 存放所有4chan图片和对应的json文件
    output_json_path = "annotate/4chan_category_results_llava.json"

    prompt = """[INST] <image>\n
    You are a harmful meme classification assistant.
    Based on the the content of the image, please select the most appropriate one or more of the following categories:
    Category list: ["religion", "politics", "race", "gender", "health status", "violence", "public health", "international relations"] 
    Please output in json format: {"category": ["category1","category2"], "confidence_score": [0.78, 0.56]}
    [/INST]"""

    # load existing results
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_ids = {item["img"] for item in results}
    else:
        processed_ids = set()
        results = []

    # get image from json
    with open(input_json_path, "r", encoding="utf-8") as f:
        image_data_list = json.load(f)

    batch_size = 10  # 设置批处理大小
    for batch in chunked(image_data_list, batch_size):
        for image_data in tqdm(batch):
            image_id = image_data["img"]

            if image_id in processed_ids:
                continue

            # url = image_data["url"]
            url = os.path.join(image_floder, image_id) # 本地加载image
            
            try:
                # image = Image.open(BytesIO(requests.get(url).content))
                image = Image.open(url).convert("RGB")
                sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=64)

                outputs = llm.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": image
                        }
                    },
                    sampling_params=sampling_params)

                generated_text = ""
                for o in outputs:
                    generated_text += o.outputs[0].text

                output_text = generated_text   
                result = extract_json(output_text)
                if result is None:
                    result = {"category": "uncertain","confidence_score": 0.0}

                result_data = {
                    "img": image_id,
                    "category": result.get("category", "uncertain"),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "raw_output": output_text
                }

            except Exception as e:
                result_data = {
                    "img": image_id,
                    "predicted_category": "error",
                    "confidence_score": 0.0,
                    "raw_output": "None",
                    "error": str(e)
                }

            results.append(result_data)
            processed_ids.add(image_id)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
        