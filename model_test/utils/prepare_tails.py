import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filepath', type=str, default='code_8.scp',help='')
parser.add_argument('--scp_savepath', type=str, default='/VAF/model_test/scp/processed_code8.scp',help='')
parser.add_argument('--trails_savepath', type=str, default='/VAF/model_test/trials/processed_code8.trials',help='')
args = parser.parse_args()

def parse_line(line):
    # /datasets/test/changjiang_longyuan_test_data/cjsd_change_voice/num_16k_real/杨瑞宾6-6.wav
    # -> sid = 杨瑞宾6-6
    sid = line.split('/')[-1].split('.')[0]
    return sid,line

if __name__ == '__main__':
    print(f"filepath: {args.filepath}")
    with open(args.filepath,'r') as f:
        lines = f.readlines()
        sids = []
        sid2wav = {}
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            sid,wav_path = parse_line(line)
            sids.append(sid)
            sid2wav[sid] = wav_path

    with open(args.scp_savepath,'w') as f:
        for sid in sids:
            f.write('{} {}\n'.format(sid,sid2wav[sid]))

    # generate trails
    with open(args.trails_savepath,'w') as f:
        for sid in sids:
            for sid2 in sids:
                if sid == sid2:
                    continue
                label = "target" if sid.split('-')[0] == sid2.split('-')[0] else "nontarget"
                f.write('{} {} {}\n'.format(sid,sid2,label))
