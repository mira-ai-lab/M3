import json
import os
import torch
from tqdm import tqdm
import pandas as pd
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_qwen2_5_vl_on_hate_meme(
    json_path: str,
    image_floder: str,
    model_name: str,
    device: str = "cuda"
):
    # 加载模型和处理器
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_name, torch_dtype="auto", device_map="auto"
    # )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

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
        # post_text = item["post_text"]
        true_label = item["label"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": f"Is this meme hateful? Answer only with 'hate' or 'normal'."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        pred_label = classify_label_from_text(output_text)

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
    df.to_csv("eval/results/binary_predictions_qwen8B_img.csv", index=False, encoding="utf-8-sig")

    # 保存整体指标到 JSON
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    with open("eval/results/binary_evaluation_qwen8B_img.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # 不再返回任何结果

if __name__ == "__main__":
    evaluate_qwen2_5_vl_on_hate_meme("dataset/CHEM.json", "dataset/img", "eval/model/Qwen3-VL-8B-Instruct")