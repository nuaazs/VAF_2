import os
import shutil

data_root_path="/ssd2/cti_aftervad_train_data_vad/"
pass_file_path="/ssd2/zs_utils/examples/ex7_calc_cos/cache/pass.txt"

fail_file_path="/ssd2/zs_utils/examples/ex7_calc_cos/cache/fail.txt"
pass_save_path="/ssd2/cti_pass_data/"
fail_save_path="/ssd2/cti_fail_data/"

def passfile_to_phone():
    f = open(pass_file_path)
    phone_num=0
    for line in f:
        line=line.strip()
        phone=line.split("/")[3]
        print(line)
        if not os.path.exists(os.path.join(pass_save_path,phone)):
            os.mkdir(os.path.join(pass_save_path,phone))
            phone_num+=1
        shutil.copy(line,os.path.join(pass_save_path,phone))
    return phone_num
def failphone_to_file():
    f = open(fail_file_path)
    file_num=0
    for phone in f:
        phone=phone.strip()
        if not os.path.exists(os.path.join(fail_save_path,phone)):
            os.mkdir(os.path.join(fail_save_path,phone))
        for filename in os.listdir(os.path.join(data_root_path,phone)):
            file_path=os.path.join(data_root_path,phone,filename)
            shutil.copy(file_path,os.path.join(fail_save_path,phone))
            file_num+=1
    return file_num
def getdirsize(dir):
   size = 0
   for root, dirs, files in os.walk(dir):
      size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
   return size
if __name__=="__main__":
    #phone_num=passfile_to_phone()
    file_num=failphone_to_file()
    size_before=getdirsize(data_root_path)
    size_after=getdirsize(pass_save_path)
    #print("phone_num",phone_num)
    print("file_num",file_num)
    #print("size_before",size_before)
    #print("size_after",size_after)
