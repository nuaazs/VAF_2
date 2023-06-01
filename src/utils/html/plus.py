# -*- coding:utf-8 -*-
import re
from utils.html.vue import BLACK_3,BLACK_2,BLACK_1,LOOKAHEAD,LOOKAHEAD_WHITE,TOTAL_WHITE,STOP_WORDS

level = {
    0: "正常",
    1: "低",
    2: "中",
    3: "高"
}

def check_filter(keywords, text):
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
    for stop_word in STOP_WORDS:
        text = text.replace(stop_word,"")
    a_list, a_count, a_text = check_filter(TOTAL_WHITE, text)
    a_text = "TOTAL_WHITE:"+a_text
    if a_count > 0:
        return None,0,f"白名单:{a_text}"
    LOOKAHEAD_WHITE_p = w2p(LOOKAHEAD_WHITE)
    a_list, a_count, a_text = check_filter(LOOKAHEAD_WHITE_p, text)
    a_text = "LOOKAHEAD_WHITE:"+a_text
    if a_count > 0:
        return None,0,f"LOOKAHEAD_WHITE:{a_text}"
    LOOKAHEAD_p = w2p(LOOKAHEAD)
    a_list, a_count, a_text = check_filter(LOOKAHEAD_p, text)
    if a_count > 0:
        return a_text,a_count,f"LOOKAHEAD:{a_text}"
    a_list, a_count, a_text = check_filter(BLACK_1, text)
    if a_count > 0:
        key_text = "black1_A:"+",".join(a_list)
        return key_text,a_count,f"BLACK_1:{a_text}"
    for key in BLACK_3.keys():
        now_dict = BLACK_3[key]
        A_list = now_dict["A"]
        B_list = now_dict["B"]
        C_list = now_dict["C"]
        a_list, a_count, a_text = check_filter(A_list, text)
        b_list, b_count, b_text = check_filter(B_list, text)
        c_list, c_count, c_text = check_filter(C_list, text)
        if a_count > 0 and b_count > 0 and c_count > 0:
            key_text = "black3_"+key+"|A:"+",".join(a_list)+"|B:"+",".join(b_list)+"|C:"+",".join(c_list)
            return key_text,a_count,key+f"_BLACK_3:{a_text},{b_text},{c_text}"
    for key in BLACK_2.keys():
        now_dict = BLACK_2[key]
        A_list = now_dict["A"]
        B_list = now_dict["B"]
        a_list, a_count, a_text = check_filter(A_list, text)
        b_list, b_count, b_text = check_filter(B_list, text)
        all_keys_text = "A:"+a_text+"| B:"+ b_text
        if "num" in now_dict.keys():
            num_th = now_dict["num"]
        else:
            num_th = 3
        if (a_count > 0 and b_count > 0) and (a_count + b_count >= num_th):
            key_text = "black2_"+key+"|A:"+",".join(a_list)+"|B:"+",".join(b_list)
            return key_text,a_count + b_count,key+f"_BLACK_2:{all_keys_text}"
    return None,0,"未命中"


def w2p(data_list):
    result_list = []
    for word_list in data_list:
        pattern = 're=^'
        for word in word_list:
            if '|' in word:
                word = f"({word})"
            keyword_regex = '(?=.*' + word + ')'
            pattern += keyword_regex
        pattern += '.+$'
        result_list.append(pattern)
    return result_list