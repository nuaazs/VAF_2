from flask import Flask, request, jsonify
import random
from dguard_nlp.interface.pretrained import load_by_name,inference

MODEL_DICT ={
    'bert_cos_b1_entropyloss':{
        "config": "/home/zhaosheng/bert_fraud_classify/text_classification/egs/easy_tc_fraud/conf/config.yaml",
        "embedding_model": '/home/zhaosheng/bert_fraud_classify/text_classification/egs/easy_tc_fraud/bert_entropyloss_simple/models/CKPT-EPOCH-17-00/embedding_model.ckpt',
        'classifier': '/home/zhaosheng/bert_fraud_classify/text_classification/egs/easy_tc_fraud/bert_entropyloss_simple/models/CKPT-EPOCH-17-00/classifier.ckpt',
    },
}
embedding_model, classifier = load_by_name('bert_cos_b1_entropyloss',device='cpu',model_dict=MODEL_DICT)
def classify_text(text):
    results = inference(embedding_model, classifier, [text], print_result=False)
    return results[0]

app = Flask(__name__)

@app.route('/text_classify', methods=['POST'])
def main():
    text = request.json['text']
    
    # 在这里添加你的分类逻辑，判断给定文本的诈骗类型
    results = classify_text(text)
    
    
    if results[0] == 0:
        type_info = "非诈骗"
        label = "0"
    else:
        if results[1] < 0.85:
            type_info = "中危"
            label = "2"
        type_info = "高危"
        label = "1"
    # print(results)
    return_data = {
        "type_info": type_info,
        "label": label,
        "score": results[1]
    }
    return jsonify(return_data)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
