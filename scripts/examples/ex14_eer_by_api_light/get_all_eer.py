import os
import numpy as np
import metrics
import cfg

if __name__ == "__main__":
    embeddings = {}
    npy_path_raw = f"../../cache/{cfg.NAME}"
    print(npy_path_raw)
    for model in cfg.MODEL_NAME.split(","):
        npy_path = os.path.join(npy_path_raw,model)
        # glob all score npy files
        score_npy_files = sorted([os.path.join(npy_path, x) for x in os.listdir(npy_path) if "scores" in x])
        labels_npy_files = sorted([os.path.join(npy_path, x) for x in os.listdir(npy_path) if "labels" in x])
        scores = []
        labels = []
        for score_npy,label_npy in zip(score_npy_files,labels_npy_files):
            print(f"Loading {score_npy}...")
            print(f"Loading {label_npy}...")
            scores += np.load(score_npy).tolist()
            labels += np.load(label_npy).tolist()
        labels = np.array(labels)
        scores = np.array(scores)
        print(f"labels: {labels.shape}")
        print(f"scores: {scores.shape}")
        # computer EER
        os.makedirs(f"./output_pngs/{cfg.NAME}/{model}",exist_ok=True)
        result = metrics.compute_eer(scores, labels,det_pic_save_path=f"./output_pngs/{cfg.NAME}/{model}/det.png",roc_pic_save_path=f"./output_pngs/{cfg.NAME}/{model}/roc.png")
        min_dcf = metrics.compute_min_dcf(result.fr, result.fa)
        print(f"EER: {result.eer}")
        print(f"minDCF: {min_dcf}")
        print(f"thresh: {result.thresh}")
        # th_list from 0.6 to 0.9, step 0.01
        th_list = np.arange(0.1,1.0,0.01)
        return_string = metrics.get_precision_reall(scores,labels,th_list)
        # write to file
        with open(f"./output_pngs/{cfg.NAME}/{model}/precision_recall.txt","w") as f:
            f.write(f"labels: {labels.shape}\n")
            f.write(f"scores: {scores.shape}\n")
            f.write(f"EER: {result.eer}\n")
            f.write(f"minDCF: {min_dcf}\n")
            f.write(f"thresh: {result.thresh}\n")
            f.write(return_string)


