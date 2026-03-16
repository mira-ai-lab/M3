from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from PIL import Image
from tqdm import tqdm
import os
import re

# Load the model and processor
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen2.5-VL/models/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen2.5-VL/models/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# prompt for the model
prompt_template = """You are a hateful meme detection assistant.
    Based on the input image and post: {post_text}, please select the most appropriate option from ["hate", "normal"].
    And give your confidence score for this option as a decimal between 0 and 1.
    Please output in this format:{"label": "hate" or "normal","confidence_score": 0.0-1.0}
"""

input_json_path = "preprocess/weibo_img_url_data.json"
image_floder = "weibo_data" #存放所有weibo图片和对应的json文件
output_json_path = "annotate/weibo_hate_results_qwen.json"

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
    json_path = os.path.join(image_floder, image_id.replace('.jpg', '.json'))
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            post_data = json.load(f)
        post_text = post_data.get("post_content", "")
    else:
        post_text = ""

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

        generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # print("generated_text:", generated_text[0])
        generated_ids_trimmed = [out_ida[len(in_ids):] for in_ids, out_ida in zip(inputs.input_ids, generated_ids)]

        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("out_text:", output_text)
        
        result = extract_json(output_text[0])
        # print("result:", result)
        if result is None:
            result = {"label": "uncertain","confidence_score": 0.0,}
        
        result_data = {
            "img": image_id,
            "label": result["label"],
            "confidence_score": result["confidence_score"],
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
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)