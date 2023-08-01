import random
import pandas as pd

# 员工信息列表
employees = [
    "张旭","胡婷","朱会","刘强","李俊贤","刘和岚","陈彩莉","吴书晨","吴洁","魏欣欣","周萍","何继凤","张浩正","侯理豪","吴琳","冯禹","周俊杰","江守兴","朱锦涛","杨飞","李君","张立斌","苏允良","王伟","祝蝶","徐行园","张羽","董帅","周海涛","王红燕","贺东海","刘单丹","杨瑞宾","张梦曲","董  乐","傅雨欣","何梦南","赵胜","齐健","薛凯翔","周坤坤","潘友健","汪泉明","吴启文","谢羽凯","李朋程","吴晶","段艺博","张龙","鲁宁","白鹭","明倩","陆晓敏","金翔","张德","杨婷姗","余子璇"
]
employees_male_add = [f"男_{i}" for i in range(1, 24)]
employees_female_add = [f"女_{i}" for i in range(1, 35)]
employees = employees + employees_male_add + employees_female_add
# 场景列表
scenes = ["公交车/站", "地铁站", "会议室", "商场", "超市", "街道", "餐馆", "酒店"]

# 语气列表
tones = ["正常", "生气", "发怒", "哀怨", "疲惫"]

# 声音大小列表
volume = ["偏大", "偏小", "正常"]

# 通话方式列表
call_type = ["手机免提", "手机非免提", "耳机"]

# 创建任务列表
tasks = []

# 分配任务给每个员工
for employee in employees:
    # 随机选择3个场景
    selected_scenes = random.sample(scenes, 3)
    
    # 随机选择2个非正常语气
    selected_tones = random.sample(tones[1:], 2)
    
    # 随机选择2个声音大小
    selected_volume = random.sample(volume, 2)
    
    # 随机选择通话方式
    selected_call_type = random.sample(call_type, 3)
    
    # 创建员工任务
    for i in range(1, 10):
        f = "男"
        if employee in ["张旭","胡婷","朱会","刘和岚","陈彩莉","吴书晨","吴洁","魏欣欣","周萍","何继凤","吴琳","李君","祝蝶","徐行园","董帅","王红燕","刘单丹","张梦曲","傅雨欣","白鹭","明倩","杨婷姗","余子璇"]:
            f= "女"
        if "女" in employee:
            f = "女"
        date_list = ["2023-07-20","2023-07-24"]
        task = [
            i,
            employee,
            f,
            random.choice(selected_scenes),
            selected_call_type[int(i%3)],
            selected_volume[i//5],
            "固话" if selected_scenes[i//3-1] == "会议室" else "手机",
            "正常" if i % 3 == 0 else random.choice(selected_tones),
            date_list[i%2]
        ]
        tasks.append(task)

# 将任务列表转换为DataFrame
df = pd.DataFrame(tasks, columns=["任务序号", "员工", "性别", "场景", "通话方式", "声音大小", "固化/手机", "语气", "录制日期"])

# 打印任务分配情况
print(df)

# 生成并保存任务分配到Excel文件
df.to_csv("task_longyuan_test_wav.csv", index=False)
