import matplotlib.pyplot as plt
import numpy as np
import os



import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument('--cache_folder', type=str, default='./cache',help='')


args = parser.parse_args()

if __name__ == '__main__':
    # load all_speaker_dict
    # merget all dicts
    # args.useful_wav_folder + '/speaker_dict_' + str(args.process_id) + '.npy'
    # find all npys
    npy_list = sorted([_file for _file in os.listdir(args.cache_folder) if "speaker_dict_" in _file and ".npy" in _file])
    # read all npys
    all_speaker_dict = {}
    dict_list = []
    for npy_file in npy_list:
        dict_list.append(np.load(args.cache_folder + '/' + npy_file, allow_pickle=True).item())
    # merge all dicts
    for dict in dict_list:
        for key in dict.keys():
            if key in all_speaker_dict.keys():
                all_speaker_dict[key] += dict[key]
            else:
                all_speaker_dict[key] = dict[key]


    # print all_speaker_dict info 
    print('all_speaker_dict info:')
    # Speaker num, average wav num, max wav num, min wav num
    print('Speaker num: ', len(all_speaker_dict.keys()))
    print('Average wav num: ', np.mean([len(all_speaker_dict[key]) for key in all_speaker_dict.keys()]))
    print('Max wav num: ', np.max([len(all_speaker_dict[key]) for key in all_speaker_dict.keys()]))
    print('Min wav num: ', np.min([len(all_speaker_dict[key]) for key in all_speaker_dict.keys()]))
    print('')
    # plot wav nums distribution
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,10))
    # plt hist from 0 to 50,bin size is 1
    plt.hist([len(all_speaker_dict[key]) for key in all_speaker_dict.keys()], bins=50, range=(0,100))
    plt.xlabel('wav nums')
    plt.ylabel('speaker nums')
    xticks = np.arange(0, 100, 2)
    plt.xticks(xticks)
    # grid
    plt.grid()
    plt.savefig(args.cache_folder + '/wav_nums_distribution.png')
    # save all_speaker_dict
    np.save(args.cache_folder + '/all_speaker_dict.npy', all_speaker_dict)
    print("Saved all_speaker_dict.npy and wav_nums_distribution.png in " + args.cache_folder)


    # Load duration_dicts and merge them    
    # find all npys
    npy_list = sorted([_file for _file in os.listdir(args.cache_folder) if "duration_cache" in _file and ".npy" in _file])
    # read all npys
    all_speaker_duration_dict = {}
    dict_list = []
    for npy_file in npy_list:
        npy_file_path = os.path.join(args.cache_folder,npy_file)
        dict_list += np.load(npy_file_path, allow_pickle=True).tolist()
    # merge all dicts
    for dict in dict_list:
        # for key in dict.keys():
        key = dict[1]
        if key in all_speaker_duration_dict.keys():
            # print(dict[key])
            all_speaker_duration_dict[key] += dict[0]
        else:
            all_speaker_duration_dict[key] = dict[0]
    # plot duration distribution
    print(all_speaker_duration_dict)
    plt.figure(figsize=(20,10))
    # plt hist
    plt.hist([all_speaker_duration_dict[key] for key in all_speaker_duration_dict.keys()])
    plt.xlabel('wav nums')
    plt.ylabel('speaker nums')
    # xticks = np.arange(0, 100, 2)
    plt.xticks([])
    # grid
    # plt.grid()
    plt.savefig(args.cache_folder + '/wav_duration_distribution.png')
    # save all_speaker_dict
    np.save(args.cache_folder + '/all_speaker_duration_dict.npy', all_speaker_duration_dict)
    print("Saved all_speaker_duration_dict.npy and wav_duration_distribution.png in " + args.cache_folder)

    

