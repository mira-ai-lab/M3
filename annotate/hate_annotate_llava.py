from io import BytesIO

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

if __name__ == "__main__":
    llm = LLM(model="LLaVA/llava-hf/llava-v1.6-mistral-7b-hf", max_model_len=4096, gpu_memory_utilization=0.95)

    input_json_path = "preprocess/4chan_img_url_data.json"
    image_floder = "4chan_data" # 存放所有4chan图片和对应的json文件
    output_json_path = "annotate/4chan_hate_results_llava.json"

    prompt = """[INST] <image>\n
    You are a hateful meme detection assistant.
    Based on the input image and post: {post_text}, please select the most appropriate option from ["hate", "normal"].
    And give your confidence score for this option as a decimal between 0 and 1.
    Please output only in exactly this format:{"label": "hate" or "normal","confidence_score": 0.0-1.0}
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

    for image_data in tqdm(image_data_list):
        image_id = image_data["img"]

        if image_id in processed_ids:
            continue

        # url = image_data["url"]
        url = os.path.join(image_floder, image_id) # 本地加载image
        json_path = os.path.join(image_floder, image_id.replace('.jpg', '.json'))
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                post_data = json.load(f)
            post_text = post_data.get("post_content", "")
        else:
            post_text = ""
        try:
            # image = Image.open(BytesIO(requests.get(url).content))
            image = Image.open(url).convert("RGB")
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)

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
                result = {"label": "uncertain","confidence_score": 0.0}

            result_data = {
                "img": image_id,
                "label": result.get("label", "uncertain"),
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