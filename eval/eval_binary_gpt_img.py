import json
import os
from tqdm import tqdm
from PIL import Image
import base64
import pandas as pd
from openai import OpenAI
from openai import BadRequestError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_gpt_on_hate_meme(
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

        base64_image = encode_image(img_path)
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": f"You are the hate speech detector. Is this meme hateful? Answer only with 'hate' or 'normal'."},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ],
            )
            output_text = completion.choices[0].message.content
            # print(f"Raw Output: {output_text}")

            pred_label = classify_label_from_text(output_text)
            
        except BadRequestError as e:
            print(f"[Warning] BadRequestError for {img_file}: {e}")
            output_text = "ERROR: Content filtered or other BadRequest"
            continue

        except Exception as e:
            print(f"[Warning] {img_file}: {e}")
            output_text = "ERROR: Other exception"
            continue
        
        # print(f"Image: {img_file}, True: {true_label}, Pred: {pred_label}, Output Text: {output_text}")

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
    df.to_csv("eval/results/binary_predictions_gpt4o_img.csv", index=False, encoding="utf-8-sig") # 改为自己的路径

    # 保存整体指标到 JSON
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    with open("eval/results/binary_evaluation_gpt4o_img.json", "w", encoding="utf-8") as f: # 改为自己的路径
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    evaluate_gpt_on_hate_meme("dataset/CHEM.json", "dataset/img") # 改为自己的路径