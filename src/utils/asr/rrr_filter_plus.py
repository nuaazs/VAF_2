# -*- coding:utf-8 -*-
import re
from utils.asr.black_3 import BLACK_3,BLACK_2,BLACK_1,LOOKAHEAD,LOOKAHEAD_WHITE,TOTAL_WHITE,STOP_WORDS

level = {
    0: "正常",
    1: "低",
    2: "中",
    3: "高"
}

def check_filter(keywords, text):
    # 模糊匹配，检查text是否包含keywords中的任意一个，打印所有匹配的结果
    # 返回匹配的结果列表和匹配的个数
    all_hit_keys = []
    for keyword in keywords:
        if "re=" in keyword:
            keyword = keyword.replace("re=","")
            matches = re.search(keyword, text)
            if matches:
                matches = matches.group(0)
                all_hit_keys.append(matches)
        if keyword in text:
            all_hit_keys.append(keyword)
    all_hit_keys = list(set(all_hit_keys))
    keys_text = ",".join(all_hit_keys)
    if len(all_hit_keys) == 0:
        keys_text = "None"
    return all_hit_keys, len(all_hit_keys), keys_text


def check_text(text):
    # 遍历BLACK_3字典的所有key，获得到诈骗类别。
    # TOTAL_WHITE
    # 如果在TOTAL_WHITE中，判白

    # 去除text中的STOP_WORDS
    for stop_word in STOP_WORDS:
        text = text.replace(stop_word,"")
    a_list, a_count, a_text = check_filter(TOTAL_WHITE, text)
    a_text = "TOTAL_WHITE:"+a_text
    if a_count > 0:
        return None,0,f"白名单:{a_text}"
    
    # LOOKAHEAD_WHITE
    # 如果在LOOKAHEAD_WHITE中，判白
    LOOKAHEAD_WHITE_p = word_list_to_pattern(LOOKAHEAD_WHITE)
    # print(LOOKAHEAD_WHITE_p)
    a_list, a_count, a_text = check_filter(LOOKAHEAD_WHITE_p, text)
    a_text = "LOOKAHEAD_WHITE:"+a_text
    if a_count > 0:
        return None,0,f"LOOKAHEAD_WHITE:{a_text}"
    
    # LOOKAHEAD
    # 如果在LOOKAHEAD中，判黑
    LOOKAHEAD_p = word_list_to_pattern(LOOKAHEAD)
    a_list, a_count, a_text = check_filter(LOOKAHEAD_p, text)
    if a_count > 0:
        return a_text,a_count,f"LOOKAHEAD:{a_text}"
    
    # BLACK_1
    a_list, a_count, a_text = check_filter(BLACK_1, text)
    if a_count > 0:
        key_text = "black1_A:"+",".join(a_list)
        return key_text,a_count,f"BLACK_1:{a_text}"

    # BLACK_2
    for key in BLACK_3.keys():
        now_dict = BLACK_3[key]
        A_list = now_dict["A"]
        B_list = now_dict["B"]
        C_list = now_dict["C"]
        # A_list B_list C_list 各自命中一次及以上，判黑
        a_list, a_count, a_text = check_filter(A_list, text)
        b_list, b_count, b_text = check_filter(B_list, text)
        c_list, c_count, c_text = check_filter(C_list, text)
        if a_count > 0 and b_count > 0 and c_count > 0:
            key_text = "black3_"+key+"|A:"+",".join(a_list)+"|B:"+",".join(b_list)+"|C:"+",".join(c_list)
            return key_text,a_count,key+f"_BLACK_3:{a_text},{b_text},{c_text}"
    
    # BLACK_2
    for key in BLACK_2.keys():
        now_dict = BLACK_2[key]
        A_list = now_dict["A"]
        B_list = now_dict["B"]
        # A_list 和 B_list 各自命中一次及以上，判黑
        a_list, a_count, a_text = check_filter(A_list, text)
        b_list, b_count, b_text = check_filter(B_list, text)
        all_keys_text = "A:"+a_text+"| B:"+ b_text
        if "num" in now_dict.keys():
            num_th = now_dict["num"]
        else:
            num_th = 3
        if (a_count > 0 and b_count > 0) and (a_count + b_count >= num_th):
            # print(a_list, b_list)
            key_text = "black2_"+key+"|A:"+",".join(a_list)+"|B:"+",".join(b_list)
            return key_text,a_count + b_count,key+f"_BLACK_2:{all_keys_text}"
        
    # LOOKAHEAD
    return None,0,"未命中"


def word_list_to_pattern(data_list):
    result_list = []
    for word_list in data_list:
        pattern = 're=^'

        for word in word_list:
            # 如果 word 中有多个关键字需要逻辑或，就用 | 替换关键字之间的空格
            if '|' in word:
                word = f"({word})"
            # 将每个单独的关键字转换为正则表达式格式
            keyword_regex = '(?=.*' + word + ')'
            # 将所有关键字拼接到一起
            pattern += keyword_regex
        pattern += '.+$'
        result_list.append(pattern)
    return result_list