from flask import Flask, request, jsonify
import random

app = Flask(__name__)

@app.route('/text_classify', methods=['POST'])
def main():
    text = request.json['text']

    # 在这里添加你的分类逻辑，判断给定文本的诈骗类型

    # 生成随机概率
    probabilities = { 
        '博彩类': round(random.uniform(0, 1), 2),
        '贷款类': round(random.uniform(0, 1), 2),
        '提现类': round(random.uniform(0, 1), 2),
        '教育退费': round(random.uniform(0, 1), 2),
        '金融类': round(random.uniform(0, 1), 2),
        '快递退费': round(random.uniform(0, 1), 2),
        '冒充公检法': round(random.uniform(0, 1), 2),
        '冒充商城售后': round(random.uniform(0, 1), 2),
        '刷单类': round(random.uniform(0, 1), 2),
        '游戏类': round(random.uniform(0, 1), 2),
        '招聘类': round(random.uniform(0, 1), 2),
        '京东金融': round(random.uniform(0, 1), 2)
    }

    # 计算概率总和
    total = sum(probabilities.values())

    # 归一化概率
    for label in probabilities:
        probabilities[label] /= total

    return jsonify(probabilities)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
