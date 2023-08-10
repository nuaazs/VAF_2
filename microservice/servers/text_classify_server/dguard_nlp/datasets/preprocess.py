
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from dguard_nlp.datasets.dataset import TCDataset
from torch.utils.data import Dataset, DataLoader

def get_dataframe(config):
    train_csv = config['train_csv']
    test_csv = config['test_csv']
    batch_size = config['batch_size']
    check_data = config['check_data']
    # df = pd.read_csv(filepath, encoding="utf8")
    # df.insert(2, 'sentence', "") 
    # for i in range(len(df)):
    #     review = df.loc[i, 'review']  # 行索引，列索引
    #     temp = re.sub('[^\u4e00-\u9fa5]+', '', review)  # 去除非汉字
    #     df.loc[i, 'sentence'] = temp
    # df = df.drop('review', axis=1)
    # df.to_csv('weibo_senti_100k_sentence.csv') # 保存

    train_df = pd.read_csv(train_csv)
    if test_csv:
        test_df = pd.read_csv(test_csv)
        X_test, y_test = test_df['sentence'].values, test_df['label'].values
        X_train,y_train = train_df['sentence'].values, train_df['label'].values
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=100)
    else:
        X_train, X_test, y_train, y_test = train_test_split(train_df['sentence'].values,
                                                        train_df['label'].values,
                                                        train_size=0.8,
                                                        random_state=100)
        # Split train to train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=100)

    train_dataset = TCDataset(X_train, y_train)
    test_dataset = TCDataset(X_test, y_test)
    vali_dataset = TCDataset(X_val, y_val)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_train[0]: {X_train[0]}, y_train[0]: {y_train[0]}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"X_test[0]: {X_test[0]}, y_test[0]: {y_test[0]}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_val[0]: {X_val[0]}, y_val[0]: {y_val[0]}")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    vali_loader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=False)
    if check_data:
        # print first 5 samples in train val and test
        for i, batch in enumerate(train_loader):
            print(f"Train data #{i}: {batch}")
            if i == 5:
                break
        for i, batch in enumerate(vali_loader):
            print(f"Vali data #{i}: {batch}")
            if i == 5:
                break
        for i, batch in enumerate(test_loader):
            print(f"Test data #{i}: {batch}")
            if i == 5:
                break
    return train_loader,vali_loader,test_loader