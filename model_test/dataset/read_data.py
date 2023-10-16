import pickle
import os

# load data_dict from file
with open('data_dict.pkl', 'rb') as f:
    data_dict_load = pickle.load(f)
print(f"Data Dict Len #{len(data_dict_load)}")

_data_info = []
speaker_list = []
for phone in data_dict_load:
    valid_size = 0
    tiny_data = []
    # print(f"Phone: {phone}")
    for fid in data_dict_load[phone]:
        
        
        if len(data_dict_load[phone][fid]) == 1 and len(phone) == 11:

            file_path = data_dict_load[phone][fid][0]
            # if file_path size < 500KB then skip
            filesize = os.path.getsize(file_path)
            
            if  filesize < 500000:
                continue
            valid_size += filesize
            tiny_data.append(f"{phone},{fid},{file_path}")
        # > 10MB
    if valid_size > 2000000*3*4:
        _data_info.extend(tiny_data)
        # print(f"Speaker: {phone}, Size: {valid_size}")
        # print(_data_info)
        speaker_list.append(phone)

speaker_list = list(set(speaker_list))
speaker_num = len(speaker_list)

# write to file
with open('data_info.txt', 'w') as f:
    for line in _data_info:
        f.write(f"{line}\n")
print(f"Speaker Num: {speaker_num}")