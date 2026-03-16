import json
import os
import torch
from tqdm import tqdm
from PIL import Image
import pandas as pd
import re
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_glm_on_hate_meme(
    json_path: str,
    image_floder: str,
    model_name: str,
    device: str = "cuda"
):
    # 加载模型和处理器
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model.eval()

    # 读取数据
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    true_labels = []
    pred_labels = []
    results_list = []

    def classify_label_from_text(text):
        text = text.lower()
        if "hate" in text:
            return "hate"
        else:
            return "normal"

    for item in tqdm(data):
        img_file = item["img"]
        img_path = os.path.join(image_floder, img_file)
        post_text = item["post_text"]
        true_label = item["label"]

        image = Image.open(img_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Below is a meme and related post.\nPost: {post_text}\nQuestion: Interpreting the meme together with the post (not the meme alone), is this meme hateful? Answer only with 'hate' or 'normal'."},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=8192)

        output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False).strip()
        answer_match = re.search(r"<answer>.*?<\|begin_of_box\|>(.*?)<\|end_of_box\|>", output_text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            # 如果提取不到，就用全输出
            answer_text = output_text
        pred_label = classify_label_from_text(answer_text)

        print(f"Image: {img_file}, True: {true_label}, Pred: {pred_label}, Output Text: {output_text}")

        results_list.append({
            "image": img_file,
            "true_label": true_label,
            "pred_label": pred_label,
            "output_text": output_text
        })

        true_labels.append(true_label)
        pred_labels.append(pred_label)

    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, pos_label="hate")
    recall = recall_score(true_labels, pred_labels, pos_label="hate")
    f1 = f1_score(true_labels, pred_labels, pos_label="hate")

    print("\n=== 评测结果 ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 保存详细预测结果到 CSV
    os.makedirs("eval/results", exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv("eval/results/binary_predictions_glm.csv", index=False, encoding="utf-8-sig") # 改为自己的路径

    # 保存整体指标到 JSON
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    with open("eval/results/binary_evaluation_glm.json", "w", encoding="utf-8") as f: # 改为自己的路径
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    evaluate_glm_on_hate_meme("dataset/CHEM.json", "dataset/img", "eval/model/GLM-4.1V-9B-Thinking") # 改为自己的路径