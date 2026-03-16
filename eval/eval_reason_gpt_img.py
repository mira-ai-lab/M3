import json
import os
from tqdm import tqdm
from PIL import Image
import base64
import pandas as pd
from openai import OpenAI
from openai import BadRequestError
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# aspect-aware BERTScore
def split_reference_aspects(reference_text):
    """
    使用分号拆分 reference explanation 中的 harm aspects
    """
    aspects = [
        asp.strip()
        for asp in reference_text.split(";")
        if len(asp.strip()) > 3
    ]
    return aspects

def aspect_aware_bertscore(
    reference_text: str,
    generated_text: str,
    threshold: float = 0.9,
    model_path: str = "eval/model/roberta-large"
):
    """
    Aspect-aware BERTScore (Recall / Coverage)
    Returns:
        coverage_score: float
        aspect_scores: list of (aspect, bert_f1)
    """
    aspects = split_reference_aspects(reference_text)
    if len(aspects) == 0:
        return 0.0, []

    covered = 0
    aspect_scores = []

    for aspect in aspects:
        _, _, F1 = bert_score(
            cands=[generated_text],
            refs=[aspect],
            model_type=model_path,
            lang="en",
            num_layers=24,
            rescale_with_baseline=False,
            device="cuda"
        )
        f1 = F1.item()
        aspect_scores.append((aspect, f1))

        if f1 >= threshold:
            covered += 1

    coverage_score = covered / len(aspects)
    return coverage_score, aspect_scores

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
        
    bleu_scores = []
    rouge_l_scores = []
    bert_f1_scores = []
    aspect_bert_scores = []
    results_list = []

    # 初始化ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for item in tqdm(data):
        img_file = item["img"]
        img_path = os.path.join(image_floder, img_file)
        post_text = item["post_text"]
        reference_explanation = item.get("reason", "").strip()
        
        base64_image = encode_image(img_path)
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"You are the hate speech explainer. Why is this meme harmful? Answer only with a verb-object phrase in English."},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ]
            )

            output_text = completion.choices[0].message.content

        except BadRequestError as e:
            print(f"[Warning] BadRequestError for {img_file}: {e}")
            continue

        except Exception as e:
            print(f"[Warning] Other exception for {img_file}: {e}")
            continue

        # BLEU
        if reference_explanation:
            reference_tokens = [reference_explanation.split()]
            hypothesis_tokens = output_text.split()
            bleu = sentence_bleu(
                reference_tokens,
                hypothesis_tokens,
                smoothing_function=SmoothingFunction().method1
            )
        else:
            bleu = 0.0

        # ROUGE-L
        if reference_explanation:
            rouge_score = scorer.score(reference_explanation, output_text)
            rouge_l_f = rouge_score["rougeL"].fmeasure
        else:
            rouge_l_f = 0.0

        # BERTScore
        if reference_explanation:
            _, _, F1 = bert_score(
                cands=[output_text],
                refs=[reference_explanation],
                model_type="eval/model/roberta-large",
                lang="en",
                num_layers=24,
                rescale_with_baseline=False,
                device="cuda"
            )
            bert_f1 = F1[0].item()
        else:
            bert_f1 = 0.0

         # ========= Aspect-aware BERTScore =========
        if reference_explanation:
            aspect_bert_coverage, aspect_bert_details = aspect_aware_bertscore(
                reference_explanation,
                output_text,
                threshold=0.9
            )
        else:
            aspect_bert_coverage = 0.0
            aspect_bert_details = []

        bleu_scores.append(bleu)
        rouge_l_scores.append(rouge_l_f)
        bert_f1_scores.append(bert_f1)
        aspect_bert_scores.append(aspect_bert_coverage)

        print(f"Image: {img_file}")
        print(f"Reference Explanation: {reference_explanation}")
        print(f"Generated Explanation: {output_text}")
        print(
            f"BLEU: {bleu:.4f}, "
            f"ROUGE-L: {rouge_l_f:.4f}, "
            f"BERTScore-F1: {bert_f1:.4f}, "
            f"Aspect-aware BERTScore: {aspect_bert_coverage:.4f}"
        )
        for asp, s in aspect_bert_details:
            print(f"  - [{s:.3f}] {asp}")
        print()

        results_list.append({
            "image": img_file,
            "reference_explanation": reference_explanation,
            "generated_explanation": output_text,
            "bleu": bleu,
            "rouge_l": rouge_l_f,
            "bert_score": bert_f1,
            "aspect_bert_coverage": aspect_bert_coverage,
            "aspect_bert_details": aspect_bert_details
        })

    # 平均指标
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    avg_bert_score = sum(bert_f1_scores) / len(bert_f1_scores) if bert_f1_scores else 0.0
    avg_aspect_bert = sum(aspect_bert_scores) / len(aspect_bert_scores) if aspect_bert_scores else 0.0

    print("\n=== 评测结果 ===")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE-L: {avg_rouge_l:.4f}")
    print(f"Average BERTScore-F1: {avg_bert_score:.4f}")
    print(f"Average Aspect-aware BERTScore: {avg_aspect_bert:.4f}")

    # 保存详细结果
    os.makedirs("eval/results", exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv("eval/results/reason_predictions_gpt4o.csv", index=False, encoding="utf-8-sig")

    # 保存整体指标
    metrics = {
        "average_bleu": avg_bleu,
        "average_rouge_l": avg_rouge_l,
        "average_bert_score": avg_bert_score,
        "average_aspect_bert": avg_aspect_bert
    }
    with open("eval/results/reason_evaluation_gpt4o.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    evaluate_gpt_on_category_multilabel(
        "dataset/CHEM_hate.json", 
        "dataset/img"
    ) 