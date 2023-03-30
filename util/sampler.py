from random import random, shuffle,randint,choice
import random as rand

def next_batch_pairwise(data,batch_size):
    '''
    full itemize pair-wise sample by batch
    '''
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            neg_item = choice(item_list)
            while neg_item in data.training_set_u[user]:
                neg_item = choice(item_list)
            j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def next_batch_pointwise(data,batch_size):
    '''
    full itemize point-wise sample by batch
    '''
    training_data = data.training_data
    data_size = len(training_data)
    batch_id = 0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y

def sample_batch_pointwise(data,batch_size):
    '''
    one batch point-wise sample, items are not interacted
    '''
    training_data = data.training_data
    data_size = len(training_data)
    idxs = [rand.randint(0,data_size-1) for i in range(batch_size)]

    users = [training_data[idx][0] for idx in idxs]
    items = [training_data[idx][1] for idx in idxs]

    u_idx, i_idx, y = [], [], []
    for i, user in enumerate(users):
        i_idx.append(data.item[items[i]])
        u_idx.append(data.user[user])
        y.append(1)
        for instance in range(4):
            item_j = randint(0, data.item_num - 1)
            while data.id2item[item_j] in data.training_set_u[user]:
                item_j = randint(0, data.item_num - 1)
            u_idx.append(data.user[user])
            i_idx.append(item_j)
            y.append(0)
    return u_idx, i_idx, y

def sample_batch_pointwise_p(data,batch_size):
    '''
    one batch point-wise sample, items can be repeated
    '''
    training_data = data.training_data
    data_size = len(training_data)
    idxs = [rand.randint(0,data_size-1) for i in range(batch_size)]

    users = [training_data[idx][0] for idx in idxs]
    items = [training_data[idx][1] for idx in idxs]

    u_idx, i_idx, y = [], [], []
    for i, user in enumerate(users):
        i_idx.append(data.item[items[i]])
        u_idx.append(data.user[user])
        y.append(1)
    return u_idx, i_idx, y


def next_batch_pointwise_1(data,batch_size):
    '''
    full itemize point-wise sample by batch
    return information in detail
    '''
    training_data = data.training_data
    data_size = len(training_data)
    batch_id = 0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, y, pos_u_idx, pos_i_idx, neg_u_idx, neg_i_idx = [], [], [], [], [], [], []

        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            pos_i_idx.append(data.item[items[i]])
            pos_u_idx.append(data.user[user])
            y.append(1)
            for instance in range(1):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                neg_u_idx.append(data.user[user])
                neg_i_idx.append(item_j)                
                y.append(0)
        yield u_idx, i_idx, y, pos_u_idx, pos_i_idx, neg_u_idx, neg_i_idx
