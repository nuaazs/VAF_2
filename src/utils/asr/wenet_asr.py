# import cfg
import requests
import cfg

# read black words
# with open(cfg.BLACK_WORDS_PATH, "r") as f:
#     black_words = f.read().splitlines()
#     black_words = set(black_words)
#     print(black_words)


def get_asr_content(wav_url="http://106.14.148.126:9000/testing/2p1c8k.wav", spkid="zhaosheng"):
    hit_keyword = False
    keyword = []
    url = cfg.ASR_SERVER
    params = {"file_url": wav_url, "spkid": spkid}
    r = requests.request("POST", url, data=params)
    if r.status_code == 200:
        text = r.json()["corrected_result"]
    else:
        text = ""
    # for word in black_words:
    #     if word in text:
    #         keyword.append(word)
    #         hit_keyword = True
    if hit_keyword:
        return text, "true", ",".join(keyword)
    else:
        return text, "false", ""

# if __name__ == '__main__':
#     result = get_asr_content()
#     print(result)
