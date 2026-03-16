from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from PIL import Image
from tqdm import tqdm
import os
import re

modal_path = "Qwen2.5-VL/models/Qwen2.5-VL-7B-Instruct"

# Load the model and processor
# default: Load the model on the available device(s)
# V100 性能会有一定损失
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    modal_path, torch_dtype="auto", device_map="auto"
) # torch.float16
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# # default processor
# processor = AutoProcessor.from_pretrained(modal_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(modal_path, min_pixels=min_pixels, max_pixels=max_pixels)

# prompt for the model
prompt_template = """ You are a harmful meme classification assistant.
    Based on the the content of the image, please select the most appropriate one or more of the following categories:
    Category list: ["religion", "politics", "race", "gender", "health status", "violence", "public health", "international relations"] 
    Please output in json format: {"category": ["category1","category2"], "confidence_score": [0.78, 0.56]}
"""

input_json_path = "preprocess/4chan_hate_img2url.json" # twitter, weibo
image_floder = "4chan_data" # 存放所有4chan图片和对应的json文件
output_json_path = "annotate/4chan_category_results_qwen.json"

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

# get json from text
def extract_json(text):
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        return None

for image_data in tqdm(image_data_list):
    image_id = image_data["img"]

    if image_id in processed_ids:
        continue

    image_path = os.path.join(image_floder, image_id)

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{image_path}"
                    },
                    {"type": "text", "text": prompt_template},
                ],
            }
        ]

        text =processor.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
       
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        # print("inputs:", inputs)

        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # print("generated_text:", generated_text[0])
        generated_ids_trimmed = [out_ida[len(in_ids):] for in_ids, out_ida in zip(inputs.input_ids, generated_ids)]

        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("out_text:", output_text)
        
        result = extract_json(output_text[0])
        # print("result:", result)
        if result is None:
            result = {"category": "uncertain","confidence_score": 0.0,}
        
        result_data = {
            "img": image_id,
            "category": result.get("category", "uncertain"),
            "confidence_score": result.get("confidence_score", 0.0),
            "raw_output": output_text[0]
        }

    except Exception as e:
        torch.cuda.empty_cache()
        result_data = {
            "img": image_id,
            "predicted_category": "error",
            "confidence_score": 0.0,
            "raw_output": "None",
            "error": str(e),
        }
    
    results.append(result_data)
    processed_ids.add(image_id)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)