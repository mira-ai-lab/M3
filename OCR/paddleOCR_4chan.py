import os
import json
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor

def process_image(filepath, ocr, fail_json_filepath):
    # 确定 JSON 文件路径
    directory, filename = os.path.split(filepath)
    json_filename = os.path.splitext(filename)[0] + ".json"
    json_filepath = os.path.join(directory, json_filename)

    try:
        # 如果 JSON 文件存在，加载内容；否则创建新数据
        if os.path.exists(json_filepath):
            with open(json_filepath, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)
                
            # 如果 recognized_text 存在且不为 null，则跳过 OCR
            # if "recognized_text" in json_data and json_data["recognized_text"] is not None:
            #     return f"Skipped: {filename}, recognized_text already exists."
        
        # OCR 识别图片
        results = ocr.ocr(filepath, cls=True)

        # 提取文字内容
        if not results[0]:
            recognized_text = None
        else:
            recognized_text = "\n".join([line[1][0] for line in results[0]])

        # 更新 JSON 文件内容
        json_data["recognized_text"] = recognized_text

        # 写入 JSON 文件
        with open(json_filepath, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)

        return f"Processed: {filename}"
    except Exception as e:
        try:
            if os.path.exists(fail_json_filepath):
                with open(fail_json_filepath, "r", encoding="utf-8") as fail_json_file:
                    fail_data = json.load(fail_json_file)
            else:
                fail_data = []

            fail_data.append({"filename": filename, "error": str(e)})

            # 写入失败记录
            with open(fail_json_filepath, "w", encoding="utf-8") as fail_json_file:
                json.dump(fail_data, fail_json_file, ensure_ascii=False, indent=4)

        except Exception as fail_e:
            return f"Failed to log error for {filename}: {fail_e}"

        return f"Failed: {filename}, Error: {e}"

def process_directory(directory, max_workers=4):
    # 初始化 PaddleOCR（全局一次初始化即可）
    ocr = PaddleOCR(use_angle_cls=True, lang="en")

    # 获取目录中的所有图片文件路径
    image_files = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith(".jpg")
    ]

    if not image_files:
        print("No image files found in the directory.")
        return

    fail_json_filepath = os.path.join(directory, "fail_4chan.json")

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, filepath, ocr, fail_json_filepath) for filepath in image_files]

        # 输出处理结果
        for future in futures:
            print(future.result())

input_directory = "4chan_data/" # 存放所有4chan图片和对应的json文件
process_directory(input_directory, max_workers=8)