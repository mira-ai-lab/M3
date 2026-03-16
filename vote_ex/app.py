from flask import Flask, render_template, request, jsonify, redirect, url_for
import json, os

app = Flask(__name__)
DATA_FILE = 'static\weibo_unreachable_images.json'
LABEL_FILE = 'weibo_user_labels.json'

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

# filtered_data = [item for item in data if item['label'][0] != item['label'][1]]
print(len(data))

def safe_load_json(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

@app.route('/')
def index():
    if os.path.exists(LABEL_FILE):
        saved = safe_load_json(LABEL_FILE)
        labeled_imgs = set(item['img'] for item in saved)
    else:
        labeled_imgs = set()

    for i, item in enumerate(data):
        if item['img'] not in labeled_imgs:
            return redirect(url_for('image', idx=i))
        
    return "All images have been labeled.", 200

@app.route('/image/<int:idx>')
def image(idx):
    if idx < 0 or idx >= len(data):
        return "No more images.", 404
    item = data[idx]
    
    is_labeled = False
    if os.path.exists(LABEL_FILE):
        saved = safe_load_json(LABEL_FILE)
        is_labeled = any(entry['img'] == item['img'] for entry in saved)

    img_path = url_for('static', filename=f'img/{item["img"]}')
    return render_template('index.html', item=item, img_path=img_path, idx=idx, total=len(data), is_labeled=is_labeled)

@app.route('/submit', methods=['POST'])
def submit():
    label_data = request.json
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, 'r') as f:
            saved = json.load(f)
    else:
        saved = []

    saved.append(label_data)

    with open(LABEL_FILE, 'w', encoding='utf-8') as f:
        json.dump(saved, f, ensure_ascii=False)

    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(port=5010, debug=True)