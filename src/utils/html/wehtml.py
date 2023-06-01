# import cfg
import requests
import cfg
from utils.html.plus import check_text

def get_html_content(wav_url="http://106.14.148.126:9000/testing/2p1c8k.wav", spkid="zhaosheng"):
    hit_keyword = False
    keyword = []
    url = cfg.ASR_SERVER
    params = {"file_url": wav_url, "spkid": spkid}
    r = requests.request("POST", url, data=params)
    if r.status_code == 200:
        text = r.json()["corrected_result"]
    else:
        text = ""
    key_word_text, key_word_count, message = check_text(text)
    if key_word_text:
        return text, "true", message
    else:
        return text, "false",message
