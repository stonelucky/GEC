import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
from sklearn.decomposition import PCA

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_dataset(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y) + 500)

    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]
    # print('adj: ', adj)
    # print('feats: ', features)
    # print('label:', labels.shape)
    return adj, features, np.argmax(labels,1)


def load_alldata(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, np.argmax(labels, 1)

def load_mat(data_str):
    data=sio.loadmat('data/'+data_str+'.mat')
    feats=data['Attributes']
    feats[feats>1] = 1
    adj=data['Network']
    label=np.reshape(data['Label'],(-1,))-1
    n_num=np.max(label)+1
    label=np.eye(n_num)[label]
    # adj = np.asarray(adj)
    # feats = np.asarray(feats)
    # np.random.seed(9102)
    # rng=np.random.get_state()
    # np.random.set_state(rng)

    # print('label:', label)

    # L=feats.shape[0]
    # ids=list(range(L))

    # val_ids=set(np.random.choice(ids,int(L*0.3),replace=False))
    # train_L=int(len(val_ids)*(1/3))
    # test_ids=sorted(list(set(ids)-val_ids))
    # train_ids=set(np.random.choice(list(val_ids),train_L,replace=False))
    # print('train_ids:', train_ids)
    # val_ids=sorted(list(val_ids-train_ids))
    # train_ids=sorted(list(train_ids))
    # print('train_ids:', train_ids)
    # y_train,y_val,y_test=np.zeros([L,n_num],dtype=np.float32),np.zeros([L,n_num],dtype=np.float32),np.zeros([L,n_num],dtype=np.float32)
    # print('label[train_ids]:',label[train_ids])
    # print('y_train[train_ids]:', y_train[train_ids])
    # y_train[train_ids]=label[train_ids]
    # y_val[val_ids]=label[val_ids]
    # y_test[test_ids]=label[test_ids]
    # train_mask,val_mask,test_mask=sample_mask(train_ids,L),sample_mask(val_ids,L),sample_mask(test_ids,L)
    return adj,feats,np.argmax(label,1)

def load_cites(data_str):
    path_cites = u"data/{}.cites".format(data_str)
    D = nx.read_edgelist(path_cites, create_using=nx.DiGraph(), nodetype=str)

    path_content = u"data/{}.content".format(data_str)
    df = pd.read_csv(filepath_or_buffer=path_content, sep='\t', header= None)
    df = df.sort_values(by=[0])
    
    adj = nx.to_scipy_sparse_matrix(D,nodelist=list(df[0].values))
    # print(type(adj))
    feature_num = len(df.columns) - 2
    feas = np.asarray(df[np.arange(1,feature_num+1)])
    # pca_3 = PCA(n_components=400)
    # feas = pca_3.fit_transform(feas)
    feas = sp.csr_matrix(feas)
    
    l = list(set(df[len(df.columns)-1]))
    mapping = {item[1]:item[0] for item in enumerate(l)}
    label = df[len(df.columns)-1].replace(mapping).values

    return adj, feas, label

def load_data(data_str):
    if data_str == 'flickr':
        return load_mat('Flickr')
    elif data_str == 'blog':
        return load_mat('BlogCatalog')
    elif data_str in ['cornell', 'texas', 'washington', 'wisconsin']:
        return load_cites(data_str)
    else:
        return load_dataset(data_str)
