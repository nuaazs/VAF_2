import  os
import csv
root='/ssd2/cti_fail_data/'

def getAllSub(path):
    Dirlist = []
    Filelist = []
    for home, dirs, files in os.walk(path):
        # 获得所有文件夹
        for dirname in dirs:
            Dirlist.append(os.path.join(home, dirname))
        # 获得所有文件
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Dirlist, Filelist
list=["手机号","文件地址","同一人","vad合格"]
csv_file = open('1.csv', 'w')
writer=csv.writer(csv_file)
writer.writerow(list)
phonelist=os.listdir(root)
phonelist.sort()
for phone in phonelist[:140]:
    kong=[]
    Dirlist,Filelist=getAllSub(os.path.join(root,phone))
    for file in Filelist:
        temp=[]
        temp.append(phone)
        temp.append(file)
        print(file)
        writer.writerow(temp)
    writer.writerow(kong)
