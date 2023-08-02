# coding = utf-8
# @Time    : 2023-07-07  08:51:25
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: .

import numpy as np
import scipy
import sklearn
from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity

try:
    import umap, hdbscan
except ImportError:
    raise ImportError(
        "Package \"umap\" or \"hdbscan\" not found. \
        Please install them first by \"pip install umap-learn hdbscan\"."
        )

from flask import Flask, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


class SpectralCluster:
    """A spectral clustering method using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, min_num_spks=1, max_num_spks=10, pval=0.02, min_pnum=6, oracle_num=None):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.min_pnum = min_pnum
        self.pval = pval
        self.k = oracle_num

    def __call__(self, X, pval=None, oracle_num=None):
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat, pval)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X):
        # Cosine similarities
        M = cosine_similarity(X, X)
        return M

    def p_pruning(self, A, pval=None):
        if pval is None:
            pval = self.pval
        n_elems = int((1 - pval) * A.shape[0])
        n_elems = min(n_elems, A.shape[0]-self.min_pnum)

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        if k_oracle is None:
            k_oracle = self.k

        lambdas, eig_vecs = scipy.linalg.eigh(L)

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1:self.max_num_spks - 1])
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        # k-means
        _, labels, _ = k_means(emb, k)
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


class UmapHdbscan:
    """
    Reference:
    - Siqi Zheng, Hongbin Suo. Reformulating Speaker Diarization as Community Detection With 
      Emphasis On Topological Structure. ICASSP2022
    """

    def __init__(self, n_neighbors=20, n_components=60, min_samples=20, min_cluster_size=10, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric

    def __call__(self, X):
        umap_X = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.0,
            n_components=min(self.n_components, X.shape[0]-2),
            metric=self.metric,
        ).fit_transform(X)
        labels = hdbscan.HDBSCAN(min_samples=self.min_samples, min_cluster_size=self.min_cluster_size).fit_predict(umap_X)
        return labels


class CommonClustering:
    """Perfom clustering for input embeddings and output the labels.
    """

    def __init__(self, cluster_type, cluster_line=10, mer_cos=None, min_cluster_size=4, **kwargs):
        self.cluster_type = cluster_type
        self.cluster_line = cluster_line
        self.min_cluster_size = min_cluster_size
        self.mer_cos = mer_cos
        if self.cluster_type == 'spectral':
            self.cluster = SpectralCluster(**kwargs)
        elif self.cluster_type == 'umap_hdbscan':
            kwargs['min_cluster_size'] = min_cluster_size
            self.cluster = UmapHdbscan(**kwargs)
        else:
            raise ValueError(
                '%s is not currently supported.' % self.cluster_type
            )

    def __call__(self, X):
        # clustering and return the labels
        assert len(X.shape) == 2, 'Shape of input should be [N, C]'
        if X.shape[0] < self.cluster_line:
            return np.ones(X.shape[0], dtype=int)
        # clustering
        labels = self.cluster(X)

        # remove extremely minor cluster
        labels = self.filter_minor_cluster(labels, X, self.min_cluster_size)
        # merge similar  speaker
        if self.mer_cos is not None:
            labels = self.merge_by_cos(labels, X, self.mer_cos)
        
        return labels
    
    def filter_minor_cluster(self, labels, x, min_cluster_size):
        cset = np.unique(labels)
        csize = np.array([(labels == i).sum() for i in cset])
        minor_idx = np.where(csize < self.min_cluster_size)[0]
        if len(minor_idx) == 0:
            return labels
        
        minor_cset = cset[minor_idx]
        major_idx = np.where(csize >= self.min_cluster_size)[0]
        major_cset = cset[major_idx]
        major_center = np.stack([x[labels == i].mean(0) \
            for i in major_cset])
        for i in range(len(labels)):
            if labels[i] in minor_cset:
                cos_sim = cosine_similarity(x[i][np.newaxis], major_center)
                labels[i] = major_cset[cos_sim.argmax()]

        return labels

    def merge_by_cos(self, labels, x, cos_thr):
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            cset = np.unique(labels)
            if len(cset) == 1:
                break
            centers = np.stack([x[labels == i].mean(0) \
                for i in cset])
            affinity = cosine_similarity(centers, centers)
            affinity = np.triu(affinity, 1)
            idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[idx] < cos_thr:
                break
            c1, c2 = cset[np.array(idx)]
            labels[labels==c2]=c1
        return labels


def process_recive_data(emb_dict):
    name_list = []
    emb_list = []
    for name, emb in emb_dict.items():
        name_list.append(name)
        emb_list.append(emb)
    emb_list = np.array(emb_list)
    r = cluster(emb_list)
    return_dict = {}
    for i in range(len(name_list)):
        return_dict[name_list[i]] = r[i]
    return return_dict


def process_receive_data(emb_dict,cluster_type,cluster_line=10,mer_cos_th=None,min_cluster_size=4):
    name_list = []
    emb_list = []
    for name, emb in emb_dict.items():
        name_list.append(name)
        emb_list.append(emb)
    emb_list = np.array(emb_list)

    # if cluster_type == "spectral":
    #     r = cluster_spectral(emb_list)
    # elif cluster_type == "umap_hdbscan":
    #     r = cluster_umap_hdbscan(emb_list)
    cluster =  CommonClustering(cluster_type, cluster_line=cluster_line, mer_cos=mer_cos_th, min_cluster_size=min_cluster_size)
    r = cluster(emb_list)
    return_dict = {}
    for i in range(len(name_list)):
        return_dict[name_list[i]] = str(r[i])
    return return_dict

def get_scores(labels_result,emb_dict):
    result = {}
    labels = []
    keys = []
    for _path in labels_result.keys():
        labels.append(labels_result[_path])
        keys.append(_path)
    _label = list(set(labels))
    _label.sort()
    for _label_item in _label:
        _label_item = str(_label_item)
        now_embeddings = []
        # print("label: ", _label_item)
        for _path in keys:
            if labels_result[_path] == _label_item and _label_item != "-1":
                now_embeddings.append(np.array(emb_dict[_path]).reshape(1,-1))
        result[_label_item] = {}
        result[_label_item]["mean"],result[_label_item]["min"],result[_label_item]["max"] = get_sim(now_embeddings)

    return result

def get_sim(vecs):
    if len(vecs) == 1:
        return 1,1,1
    sims = []
    for x_index in range(len(vecs)):
        for y_index in range(len(vecs)):
            if x_index == y_index:
                continue
            vec1 = vecs[x_index]
            vec2 = vecs[y_index]
            sims.append(cosine_similarity(vec1,vec2))
            
    return np.mean(sims),np.min(sims),np.max(sims)

@app.route('/cluster', methods=['POST'])
def cluster():
    input_data = request.get_json()
    emb_dict = input_data['emb_dict']
    cluster_line = input_data.get('cluster_line', 10)
    mer_cos_type = input_data.get('mer_cos_th', 0.8)
    cluster_type = input_data.get('cluster_type', "spectral")
    min_cluster_size = input_data.get('min_cluster_size', 4)

    
    return_data = {}
    result = process_receive_data(emb_dict,cluster_type,cluster_line,mer_cos_type,min_cluster_size)
    # print(result)
    scores = get_scores(result,emb_dict)
    return_data["labels"] = result
    return_data["scores"] = scores
    print(return_data)
    return jsonify(return_data)

if __name__ == '__main__':

    # cluster_spectral =  CommonClustering("spectral", cluster_line=cluster_line, mer_cos=mer_cos_th, min_cluster_size=min_cluster_size)
    # cluster_umap_hdbscan =  CommonClustering("umap_hdbscan", cluster_line=cluster_line, mer_cos=mer_cos_th, min_cluster_size=min_cluster_size)
    app.run(host='0.0.0.0', port=5011)

# if __name__ == "__main__":
#     # test
#     cluster =  CommonClustering('spectral', cluster_line=20,mer_cos=None,min_cluster_size=4)
#     emb_dict = {"filename1":np.random.rand(192).tolist(),"filename2":np.random.rand(192).tolist(),"filename3":np.random.rand(192).tolist()}
#     print(process_recive_data(emb_dict))