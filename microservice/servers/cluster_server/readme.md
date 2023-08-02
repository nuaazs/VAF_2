# API文档

## 聚类接口

### 接口描述

该接口用于将输入的嵌入向量进行聚类。

### 请求URL

- `http://localhost:5011/cluster`

### 请求方法

- POST

### 请求参数

| 参数名           | 类型   | 描述                    |
|------------------|--------|-------------------------|
| emb_dict         | dict   | 嵌入向量字典，键为文件名，值为嵌入向量列表     |
| cluster_line     | int    | 聚类数量      |
| mer_cos_th       | float  | 相似度阈值，大于该值的嵌入向量将被合并为同一簇   |
| cluster_type     | str    | 聚类方法类型，可选值为 "spectral" 或 "umap_hdbscan" |
| min_cluster_size | int    | 最小簇大小，小于该值的簇将被过滤掉            |

### 请求示例

```json
{
    "emb_dict": {
        "filename1": [0.1, 0.2, 0.3],
        "filename2": [0.4, 0.5, 0.6],
        "filename3": [0.7, 0.8, 0.9]
    },
    "cluster_line": 10,
    "mer_cos_th": 0.8,
    "cluster_type": "spectral",
    "min_cluster_size": 4
}
```

### 返回结果

返回结果为一个字典，键为文件名，值为对应的聚类标签。

### 返回示例

```json
{
    "filename1": 0,
    "filename2": 1,
    "filename3": 0
}
```
