import numpy as np
import pandas as pd
import re

data_set = pd.read_csv('data/data.csv').values
train_set = set(pd.read_csv('data/train.csv').values.flatten())
test_set = set(pd.read_csv('data/test.csv').values.flatten())
val_set = set(pd.read_csv('data/val.csv').values.flatten())
length = len(data_set)

def data_analysis():
    data_group = []
    Entire, Private, Shared, Others= classifier()
    for array in [Entire, Private, Shared, Others]:
        train_data, val_data, test_data = [], [], []
        Id = np.array([int(i) for i in array[:,0]])
        host, properties, amenitie = host_reply(array), longtext(array), longtext(array)
        accommodate, bedroom, bed, bathroom = data_numbers(array[:,13]), data_numbers(array[:,15]), data_numbers(array[:,16]), data_numbers(array[:,14])
        review_ltm, review, review_rating, review_accuracy = data_numbers(array[:,21]), data_numbers(array[:,20]), data_numbers(array[:,24]), data_numbers(array[:,25])
        communication, clean, checkin, location = data_numbers(array[:,28]), data_numbers(array[:,26]), data_numbers(array[:,27]), data_numbers(array[:,29])
        value, price = data_numbers(array[:,30]), data_numbers(array[:,31])
        t_data = list(np.c_[Id, np.ones(len(array)), host, properties, accommodate, bathroom, bedroom, bed, amenitie, review, review_ltm, review_rating, review_accuracy, clean, checkin, communication, location, value, price])
        for obj in t_data:
            if obj[0] in train_set:
                train_data.append(obj[1:])
            elif obj[0] in val_set:
                val_data.append(obj[1:])
            elif obj[0] in test_set:
                test_data.append(obj[:-1])
        data_group.append([np.array(train_data), np.array(val_data), np.array(test_data)])
    return data_group[0], data_group[1], data_group[2], data_group[3]

def longtext(array):
    label, lists, json = [], set([]), array[:, 18]
    for s in json:
        strings = re.split('{|"|,|}', s)
        texttrue = []
        for t in strings:
            if t != "":
                texttrue.append(t)
                lists.add(t)
        label += [texttrue]
    lists, len_list = list(lists), len(lists)
    postion, result = {}, []
    for i in range(len_list):
        postion[lists[i]] = i
    for i in label:
        t = [0]*len_list
        for j in i:
            t[postion[j]] = 1.0
        result.append(t)
    return np.array(result)

def data_numbers(array):
    l, result, nan, total, sums = len(array), [], 0, [], 0
    for i in range(l):
        s = array[i]
        if s != 'nan': 
            t1 = float(s)
            result.append(t1)
            nan += 1
            sums += t1
        else:
            total.append(i)
            result.append(0)
    avg = sums/nan
    for i in total:
        result[i] = avg
    return np.array(result)

def host_reply(array):
    text, result, l = {'a few days or more':0, 'within a day':0.1, 'within a few hours':0.2, 'within an hour':0.3, 't':1.0, 'f':0.0},[], len(array)
    response, rate, spl, identity = array[:, 2], array[:, 3], array[:, 4], array[:, 6]
    n, st, sr, ss, ns = 0, 0, 0, 0, 0
    for i in range(l):
        if spl[i] == 't':
            ss += 1
            ns += 1
        if response[i] in text:
            n += 1
            pts = text[response[i]]
            pst = float(rate[i][:-1])
            result.append([pts, pst, text[spl[i]]])
            st += pts
            sr += pst
        else:
            result.append(-1)
    at, ar, ass = st/n, sr/n, ss/ns
    for i in range(l):
        if result[i] == -1:
            if type(spl[i]) == str:
                result[i] = [at, ar, drn[spl[i]]]
            else:
                result[i] = [at, ar, ass]
    for i in range(l):
        if identity[i] == 't':
            result[i].append(1.0)
        else:
            result[i].append(0.0)
    return np.array(result)

def classifier():
    Entire, Private, Shared, Others = [], [], [], []
    for i in range(length):
        obj = data_set[i]
        room = obj[12]
        obj = list(obj)
        if room == 'Entire home/apt':
            Entire.append(obj)
        elif room == 'Private room':
            Private.append(obj)
        elif room == 'Shared room':
            Shared.append(obj)
        else:
            Others.append(obj)
    return np.array(Entire), np.array(Private), np.array(Shared), np.array(Others)


Entire, Private, Shared, Others = data_analysis()
TEntire, TyE, TPrivate, TyP = Entire[0][:,:-1], Entire[0][:,-1], Private[0][:,:-1], Private[0][:,-1]
TShared, TyS, TOthers, TyO =  Shared[0][:,:-1], Shared[0][:,-1], Others[0][:,:-1], Others[0][:,-1]
bE, bP, bS, bO = np.dot(np.dot(np.linalg.pinv(np.dot(TEntire.T, TEntire)), TEntire.T), TyE), np.dot(np.dot(np.linalg.pinv(np.dot(TPrivate.T, TPrivate)), TPrivate.T), TyP), np.dot(np.dot(np.linalg.pinv(np.dot(TShared.T, TShared)), TShared.T), TyS), np.dot(np.dot(np.linalg.pinv(np.dot(TOthers.T, TOthers)), TOthers.T), TyO)
VEntire, VyE, VPrivate, VyP, VShared, VyS, VOthers, VyH = Entire[1][:,:-1], Entire[1][:,-1], Private[1][:,:-1], Private[1][:,-1], Shared[1][:,:-1], Shared[1][:,-1], Others[1][:,:-1], Others[1][:,-1]
VpE, VpP, VpS, VpO = np.dot(VEntire, bE), np.dot(VPrivate, bP), np.dot(VShared, bS), np.dot(VOthers, bO)
TeEntire, TePrivate, TeShared, TeOthers = Entire[2][:,1:], Private[2][:,1:], Shared[2][:,1:], Others[2][:,1:]
idType = [Entire[2][:,0], Private[2][:,0], Shared[2][:,0], Others[2][:,0]]
pType = [np.dot(TeEntire, bE), np.dot(TePrivate, bP), np.dot(TeShared, bS), np.dot(TeOthers, bO)]

predicted = {}
for i in range(len(idType)):
    ids = idType[i]
    price = pType[i]
    for j in range(len(ids)):
        predicted[int(ids[j])] = round(price[j])
result = []

for i in list(pd.read_csv('data/test.csv').values.flatten()):
    result.append([i, predicted[i]])
    
save = pd.DataFrame(result, columns = ['id', 'price'])
save.to_csv('pred.csv',index=False,header=True)
