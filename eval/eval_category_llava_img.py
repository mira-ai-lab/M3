import json
import os
import torch
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score

# 全局标签列表
CATEGORY_LABELS = ["religion", "politics", "race", "gender", 
                   "health status", "violence", "public health", "international relations"]

def evaluate_llava_on_category_multilabel(
    json_path: str,
    image_floder: str,
    model_name: str,
    device: str = "cuda"
):
    # 加载模型和处理器
    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model.to(device)

    # 读取数据
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
        img_path = os.path.join(image_floder, img_file)
        post_text = item["post_text"]
        true_category = item["category"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_path},
                    {"type": "text", "text": f"Below is a meme and related post.\nPost: {post_text}\nQuestion: Which categories does the harmfulness of this meme? The possible categories are: {', '.join(CATEGORY_LABELS)}. Answer ONLY with a list of category names, no explanation."},
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
            generated_ids = model.generate(**inputs, max_new_tokens=32)

        output_text = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        output_text = output_text.split("ASSISTANT:")[-1].strip()
        
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

    # 保存详细预测结果到 CSV
    os.makedirs("eval/results", exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv("eval/results/category_predictions_llava13B_img.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "hamming_loss": hl,
        "subset_accuracy": subset_acc
    }
    with open("eval/results/category_evaluation_llava13B_img.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    evaluate_llava_on_category_multilabel(
        "dataset/CHEM_hate.json", 
        "dataset/img", 
        "eval/model/llava-v1.6-vicuna-13b-hf"
    ) 