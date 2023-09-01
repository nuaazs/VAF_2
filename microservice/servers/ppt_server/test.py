import datetime
import random
import string
import pymysql
from faker import Faker
import cfg
import phone as ph

msg_db = cfg.MYSQL

# 创建 Faker 对象
faker = Faker(locale='zh_CN')

def faker_black():
    # 连接MySQL数据库
    conn = pymysql.connect(
            host=msg_db.get("host"),
            port=msg_db.get("port"),
            db=msg_db.get("db"),
            user=msg_db.get("username"),
            passwd=msg_db.get("passwd"),
            cursorclass=pymysql.cursors.DictCursor,
        )
    cursor = conn.cursor()
    # 生成并插入100条数据
    for _ in range(100):
        name = faker.name()
        gender = random.choice(['男', '女'])
        phone = faker.phone_number()
        try:
            info = ph.Phone().find(phone)
            phone_area = info['province'] + "-" + info['city']
        except Exception as e:
            phone_area = ""
        record_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
        phone_area = ""
        file_url = faker.url()
        preprocessed_file_url = faker.url()
        valid_length = random.randint(10, 20)
        call_begintime = faker.date_time_this_year()
        call_endtime = call_begintime + datetime.timedelta(seconds=valid_length)
        register_time = faker.date_time_this_month()
        record_month =random.randint(5, 8)
        record_type = ""
        spectrogram_url = faker.url()

        with conn.cursor() as cursor:
            sql = "INSERT INTO si.black_speaker_info_ppt (name, gender, phone, record_id, phone_area, file_url, preprocessed_file_url, valid_length, call_begintime, call_endtime, register_time, record_month, record_type, spectrogram_url) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (name, gender, phone, record_id, phone_area, file_url, preprocessed_file_url, valid_length, call_begintime, call_endtime, register_time, record_month, record_type, spectrogram_url))

    conn.commit()
    conn.close()


def faker_hit():
    conn = pymysql.connect(
            host=msg_db.get("host"),
            port=msg_db.get("port"),
            db=msg_db.get("db"),
            user=msg_db.get("username"),
            passwd=msg_db.get("passwd"),
            cursorclass=pymysql.cursors.DictCursor,
        )
    cursor = conn.cursor()
    for _ in range(100):
        name = faker.name()
        phone = faker.phone_number()
        gender = random.choice(['男', '女'])
        valid_length = random.randint(60, 600)
        file_url = faker.url()
        preprocessed_file_url = faker.url()
        phone_type = random.choice(['智能手机', '座机', '其他'])
        phone_area = ""
        call_begintime = faker.date_time_this_year()
        call_endtime = call_begintime + datetime.timedelta(seconds=valid_length)
        class_number = random.randint(1, 10)
        hit_time = faker.date_time_this_month()
        hit_score = round(random.uniform(0, 1), 2)
        hit_spkid = faker.phone_number()
        model02_blackbase_phone = faker.uuid4()
        eres2net_hit_score = f"{round(random.uniform(0, 1), 2)},{round(random.uniform(0, 1), 2)}"
        model02_hit_score = random.randint(1, 10)
        eres2net_top_10 = ','.join([faker.word() for _ in range(10)])
        model02_top_10 = ','.join([faker.word() for _ in range(10)])
        double_check = faker.text()
        gender_score = round(random.uniform(0, 1), 2)
        message = faker.sentence()

        with conn.cursor() as cursor:
            sql = "INSERT INTO si.hit_ppt (name, phone, gender, valid_length, file_url, preprocessed_file_url, phone_type, phone_area, call_begintime, call_endtime, class_number, hit_time, hit_score, hit_spkid, model02_blackbase_phone, eres2net_hit_score, model02_hit_score, eres2net_top_10, model02_top_10, double_check, gender_score, message) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (name, phone, gender, valid_length, file_url, preprocessed_file_url, phone_type, phone_area, call_begintime, call_endtime, class_number, hit_time, hit_score, hit_spkid, model02_blackbase_phone, eres2net_hit_score, model02_hit_score, eres2net_top_10, model02_top_10, double_check, gender_score, message))

    conn.commit()
    conn.close()


# faker_hit()
# faker_black()


data = {
    "item1": {
        "key_to_remove": "value1",
        "other_key": "other_value"
    },
    "item2": {
        "key_to_remove": "value2",
        "other_key": "other_value"
    }
}

key_to_remove = "key_to_remove"

del data['item1'][key_to_remove]

print(data)
