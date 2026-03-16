import json
import os
import torch
from tqdm import tqdm
import pandas as pd
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score

# 全局标签列表
CATEGORY_LABELS = ["religion", "politics", "race", "gender", 
                   "health status", "violence", "public health", "international relations"]

def evaluate_qwen2_5_vl_on_category_multilabel(
    json_path: str,
    image_folder: str,
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

    # 加载数据
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    true_labels = []
    pred_labels = []
    results_list = []

    def parse_categories_from_output(text):
        """
        假设模型输出: "['politics', 'race']" 或 "politics, race"
        返回 ['politics', 'race']
        """
        text = text.lower()
        for ch in ['[', ']', "'", '"']:
            text = text.replace(ch, '')
        parts = [t.strip() for t in text.split(",")]
        # 只保留在标签列表中的
        return [p for p in parts if p in CATEGORY_LABELS]

    for item in tqdm(data):
        img_file = item["img"]
        img_path = os.path.join(image_folder, img_file)
        post_text = item["post_text"]
        true_category = item["category"]  # 多标签 list

        # 提问：请只回答标签列表
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": f"Below is a meme and related post.\nPost: {post_text}\nQuestion: Interpreting the meme together with the post (not the meme alone), which categories does the harmfulness of this meme? The possible categories are: {', '.join(CATEGORY_LABELS)}. Answer ONLY with a list of category names, no explanation."},
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
            generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        pred_category = parse_categories_from_output(output_text)

        print(f"Image: {img_file}")
        print(f"True categories: {true_category}")
        print(f"Pred categories: {pred_category}")
        print(f"Model raw output: {output_text}\n")

        # 转 multi-hot
        true_labels.append([1 if label in true_category else 0 for label in CATEGORY_LABELS])
        pred_labels.append([1 if label in pred_category else 0 for label in CATEGORY_LABELS])

        results_list.append({
            "image": img_file,
            "true_category": true_category,
            "pred_category": pred_category,
            "output_text": output_text
        })

    # 计算指标
    macro_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    macro_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    hl = hamming_loss(true_labels, pred_labels)
    subset_acc = accuracy_score(true_labels, pred_labels)

    print("\n=== Category 多标签评测结果 ===")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Hamming Loss:   {hl:.4f}")
    print(f"Subset Accuracy:{subset_acc:.4f}")

    # 保存结果
    os.makedirs("eval/results", exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv("eval/results/category_predictions_qwen8B.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "hamming_loss": hl,
        "subset_accuracy": subset_acc
    }
    with open("eval/results/category_evaluation_qwen8B.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    evaluate_qwen2_5_vl_on_category_multilabel(
        "dataset/CHEM_hate.json",
        "dataset/img",
        "eval/model/Qwen3-VL-8B-Instruct"
    )