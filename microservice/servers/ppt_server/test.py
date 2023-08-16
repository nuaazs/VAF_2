import random
import pymysql
from faker import Faker
from datetime import datetime, timedelta

import cfg

# 初始化Faker实例
fake = Faker('zh_CN')
msg_db = cfg.MYSQL

# 数据库连接初始化
def get_db_connection():
    return    pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )

# 生成并插入100条假数据
def generate_and_insert_fake_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    for _ in range(100):
        name = fake.name()
        gender = fake.random_element(['M', 'F'])
        phone = fake.phone_number()
        record_id = fake.uuid4()
        phone_area = random.choice(['江苏-南京', '江苏-镇江', '江苏-徐州', '江苏-宿迁'])
        file_url = "http://192.168.3.169:9000/raw/15720786010/raw_15720786010.wav"
        preprocessed_file_url = "http://192.168.3.169:9000/raw/15720786010/15720786010_selected.wav"
        valid_length = fake.random_int(min=1, max=60)
        call_begintime = fake.date_time_between(start_date='-1y', end_date='now')
        call_endtime = call_begintime + timedelta(minutes=valid_length)
        register_time = fake.date_time_between(start_date='-1y', end_date='now')
        record_month = "8"
        record_type = fake.random_element(['incoming', 'outgoing'])
        spectrogram_url = ""

        query = '''
        INSERT INTO black_speaker_info_ppt (name, gender, phone, record_id, phone_area, file_url,
        preprocessed_file_url, valid_length, call_begintime, call_endtime, register_time, record_month,
        record_type, spectrogram_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''

        values = (name, gender, phone, record_id, phone_area, file_url, preprocessed_file_url,
                  valid_length, call_begintime, call_endtime, register_time, record_month,
                  record_type, spectrogram_url)

        cursor.execute(query, values)
        conn.commit()
    cursor.close()
    conn.close()


def insert_fake_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    for _ in range(10):
        name = fake.name()
        phone = fake.phone_number()
        gender = fake.random_element(['M', 'F'])
        valid_length = fake.random_int(min=1, max=60)
        file_url = fake.url()
        preprocessed_file_url = fake.url()
        phone_type = fake.random_element(['Android', 'iOS'])
        phone_area = fake.random_element(['Beijing', 'Shanghai', 'Guangzhou'])
        call_begintime = fake.date_time_between(start_date='-1y', end_date='now')
        call_endtime = call_begintime + timedelta(minutes=valid_length)
        class_number = fake.random_int(min=1, max=10)
        hit_time = fake.date_time_between(start_date=call_begintime, end_date=call_endtime)
        hit_score = fake.pyfloat(left_digits=3, right_digits=2, positive=True)
        hit_spkid = fake.random_element(['spk001', 'spk002', 'spk003'])
        model02_blackbase_phone = fake.random_element(['black001', 'black002', 'black003'])
        eres2net_hit_score = fake.random_element(['score001', 'score002', 'score003'])
        model02_hit_score = fake.random_int(min=1, max=100)
        eres2net_top_10 = fake.random_element(['top001', 'top002', 'top003'])
        model02_top_10 = fake.random_element(['top001', 'top002', 'top003'])
        double_check = fake.random_element(['check001', 'check002', 'check003'])
        gender_score = fake.pyfloat(left_digits=3, right_digits=2, positive=True)
        message = fake.sentence()

        query = '''
        INSERT INTO hit_ppt (name, phone, gender, valid_length, file_url, preprocessed_file_url,
        phone_type, phone_area, call_begintime, call_endtime, class_number, hit_time, hit_score,
        hit_spkid, model02_blackbase_phone, eres2net_hit_score, model02_hit_score, eres2net_top_10,
        model02_top_10, double_check, gender_score, message)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''

        values = (name, phone, gender, valid_length, file_url, preprocessed_file_url, phone_type,
                  phone_area, call_begintime, call_endtime, class_number, hit_time, hit_score,
                  hit_spkid, model02_blackbase_phone, eres2net_hit_score, model02_hit_score,
                  eres2net_top_10, model02_top_10, double_check, gender_score, message)

        cursor.execute(query, values)
        conn.commit()

    cursor.close()
    conn.close()


if __name__ == '__main__':
    generate_and_insert_fake_data()
