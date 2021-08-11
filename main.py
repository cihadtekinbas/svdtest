import pandas as pd
import numpy as np
data = pd.read_csv('Aratings-transform.csv')
data['userId'] = data['userId'].astype('str')
data['movieId'] = data['movieId'].astype('str')
users = data['userId'].unique() #list of all users
movies = data['movieId'].unique() #list of all moviesprint("Number of users", len(users))

users = data['userId'].unique() #list of all users
movies = data['movieId'].unique() #list of all movies

train = data



data = pd.read_csv('Bratings-transform.csv')
data['userId'] = data['userId'].astype('str')
data['movieId'] = data['movieId'].astype('str')
users = data['userId'].unique() #list of all users
movies = data['movieId'].unique() #list of all moviesprint("Number of users", len(users))

users = data['userId'].unique() #list of all users
movies = data['movieId'].unique() #list of all movies

test = data

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
def create_utility_matrix(data, formatizer = {'user':0, 'item': 1, 'value': 2}):
    itemField = formatizer['item']
    userField = formatizer['user']
    valueField = formatizer['value']
    
    userList = data.iloc[:,userField].tolist()
    itemList = data.iloc[:,itemField].tolist()
    valueList = data.iloc[:,valueField].tolist()
    
    users = list(set(data.iloc[:,userField]))
    items = list(set(data.iloc[:,itemField]))
    
    users_index = {users[i]: i for i in range(len(users))}
    
    pd_dict = {item: [np.nan for i in range(len(users))] for item in items}
    
    for i in range(0,len(data)):
        item = itemList[i]
        user = userList[i]
        value = valueList[i]
    
    
    pd_dict[item][users_index[user]] = value
    
    X = pd.DataFrame(pd_dict)
    X.index = users
    
    itemcols = list(X.columns)
    items_index = {itemcols[i]: i for i in range(len(itemcols))}
    
    return X, users_index, items_index

def svd(train, k):
    utilMat = np.array(train)
    
    mask = np.isnan(utilMat)
    masked_arr = np.ma.masked_array(utilMat, mask)
    item_means = np.mean(masked_arr, axis=0)
    
    utilMat = masked_arr.filled(item_means)
    
    x = np.tile(item_means, (utilMat.shape[0],1))
    
    utilMat = utilMat - x
    
    U, s, V=np.linalg.svd(utilMat, full_matrices=False)
    s=np.diag(s)
    
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]
    
    s_root=sqrtm(s)
    
    Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)
    UsV = np.dot(Usk, skV)
    
    UsV = UsV + x
    print("svd done")
    return UsV


# from recsys import svd, create_utility_matrix


def rmse(true, pred):
    # this will be used towards the end
    x = true - pred
    return sum([xi*xi for xi in x])/len(x)

no_of_features = [8,10,12,14,17]

utilMat, users_index, items_index = create_utility_matrix(train)


for f in no_of_features: 
    svdout = svd(utilMat, k=f)
    pred = [] #to store the predicted ratings
    
    for _,row in test.iterrows():
        user = row['userId']
        item = row['movieId']
        
        
        u_index = users_index[user]
        if item in items_index:
            i_index = items_index[item]
            pred_rating = svdout[u_index, i_index]
        else:
            pred_rating = np.mean(svdout[u_index, :])
        pred.append(pred_rating)
        
        
    print(rmse(test['rating'], pred))
