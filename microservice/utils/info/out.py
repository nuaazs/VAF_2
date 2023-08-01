from utils import preprocess
from utils.log import err_logger
from utils.log import logger
import datetime

ERR_TYPE = {
    0: "Okay.",
    1: "The file format is incorrect",
    2: "File parsing error.",
    3: "File download error.",
    4: "Speaker already exist.",
    5: "预处理报错",
    6: "VAD报错",
    7: "声纹提取错误",
    8: "聚类错误",
    9: "上传文件错误",
    10: "Request error.",
    12: "性别识别失败",
}


class OutInfo:
    """
    out dict info
    """

    used_time = {
        # "download_used_time": 0,
        # "vad_used_time": 0,
        # "classify_used_time": 0,
        # "embedding_used_time": 0,
        # "self_test_used_time": 0,
        # "to_database_used_time": 0,
        # "test_used_time": 0,
    }

    def __init__(self):
        action_type="test"
        self.response = {"code": 2000, "status": "error"}
        self.action_type = action_type
        self.response_check_new = {
            "code": 2000,
            "status": "error",
            "err_type": "None",
            "err_msg": "None",
            "inbase": True,
            "code": 2000,
            "replace": False,
        }
        self.start_time = datetime.datetime.now()
        self.oss_path = ""
        self.preprocess_file_path = ""
        self.show_phone = ""
        self.before_vad_length = 0
        self.after_vad_length = 0
        self.wav = ""
        self.wav_vad = ""
        self.embedding = ""
        self.embeddings_name_list=[]
        self.embeddings_list=[]
        self.embeddings_dict={}
        self.class_num = ""
        self.call_begintime = ""
        self.call_endtime = ""
        self.self_test_result = ""
        self.raw_minio_file_url = ""
        self.gender_result = {}


    def log_time(self, name):
        self.used_time[name] = (datetime.datetime.now() - self.start_time).total_seconds()
        self.start_time = datetime.datetime.now()

    def response_error(self, spkid, err_type, message="", used_time=None, show_phone=None):
        if show_phone != None:
            self.show_phone = show_phone
        else:
            self.show_phone = spkid

        err_msg = ERR_TYPE[err_type]
        err_msg = err_msg.replace("'", "")
        message = message.replace("'", "")
        err_logger.error(f"{spkid},{self.oss_path},{err_type},{err_msg},{message}")
        logger.error(f"{spkid},{self.oss_path},{err_type},{err_msg},{message}")
        self.response["err_type"] = err_type
        self.response["err_msg"] = err_msg
        self.response["message"] = message
        if used_time != None:
            self.response["used_time"] = used_time
        return self.response

    def response_vad(self, err_type, err_msg, used_time=None):
        self.response["err_type"] = err_type
        self.response["err_msg"] = err_msg
        if used_time != None:
            self.response["used_time"] = used_time
        return self.response

    def response_check(self, err_type, err_msg):
        self.response_check_new["err_type"] = err_type
        self.response_check_new["err_msg"] = err_msg
        return self.response
