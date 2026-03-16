import json
import os
from tqdm import tqdm
from PIL import Image
import base64
import pandas as pd
from openai import OpenAI
from openai import BadRequestError
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score

# 全局标签列表
CATEGORY_LABELS = ["religion", "politics", "race", "gender", 
                   "health status", "violence", "public health", "international relations"]

def evaluate_gpt_on_category_multilabel(
    json_path: str,
    image_floder: str
):
    # 设置 API key 和 API base URL
    api_key = ""
    base_url = ""

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 读取数据
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    true_labels = []
    pred_labels = []
    results_list = []

    def parse_categories_from_output(text):
        """
        假设模型输出: "['politics', 'race']" 或 "politics, race"
        返回 ['politics', 'race']
        """
        text = text.lower()
        for ch in ['[', ']', "'", '"', '.']:
            text = text.replace(ch, '')
        parts = [t.strip() for t in text.split(",")]
        # 只保留在标签列表中的
        return [p for p in parts if p in CATEGORY_LABELS]

    for item in tqdm(data):
        img_file = item["img"]
        img_path = os.path.join(image_floder, img_file)
        post_text = item["post_text"]
        true_category = item["category"]  # 多标签 list

        base64_image = encode_image(img_path)
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Which categories does the harmfulness of this meme? The possible categories are: {', '.join(CATEGORY_LABELS)}. Answer ONLY with a list of category names, no explanation."},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ]
            )

            output_text = completion.choices[0].message.content
            pred_category = parse_categories_from_output(output_text)

        except BadRequestError as e:
            print(f"[Warning] BadRequestError for {img_file}: {e}")
            output_text = "ERROR: Content filtered or other BadRequest"
            continue

        except Exception as e:
            print(f"[Warning] Other exception for {img_file}: {e}")
            output_text = "ERROR: Other exception"
            continue

        # print(f"Image: {img_file}")
        # print(f"True categories: {true_category}")
        # print(f"Pred categories: {pred_category}")
        # print(f"Model raw output: {output_text}\n")

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
    df.to_csv("eval/results/category_predictions_gpt4o_img.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "hamming_loss": hl,
        "subset_accuracy": subset_acc
    }
    with open("eval/results/category_evaluation_gpt4o_img.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    evaluate_gpt_on_category_multilabel(
        "dataset/CHEM_hate.json", 
        "dataset/img"
    ) 