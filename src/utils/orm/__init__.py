# coding = utf-8
# @Time    : 2022-09-05  15:05:48
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: ORM.

from utils.orm.database import get_embeddings
from utils.orm.database import to_database
from utils.orm.database import delete_by_key
from utils.orm.query import check_url
from utils.orm.query import check_spkid
from utils.orm.query import to_log
from utils.orm.query import add_hit
from utils.orm.query import add_hit_count
from utils.orm.query import add_speaker
from utils.orm.query import get_spkinfo
from utils.orm.query import get_blackid
from utils.orm.query import delete_spk
