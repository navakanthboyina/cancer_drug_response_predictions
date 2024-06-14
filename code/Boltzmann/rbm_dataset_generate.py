from utils.util import *
import math

(X_train,Y_train,X_valid,Y_valid,cell2id_mapping,drug2id_mapping) = prepare_train_data("../data/drugcell_train.txt","../data/drugcell_val.txt","../data/cell2ind.txt","../data/drug2ind.txt")
drug_fingerprint = pd.read_csv("../data/drug2fingerprint.txt",header=None)
numpy_arr = torch.from_numpy(drug_fingerprint.to_numpy().astype(np.float32))
#print(numpy_arr)
cell_mutation =  pd.read_csv("../data/cell2mutation.txt",header=None)
cell_arr = torch.from_numpy(cell_mutation.to_numpy().astype(np.float32))
drug_encodings = get_encodings_rbm_vae("../model/rbm_drug_50",numpy_arr)
cell_encodings = get_encodings_rbm_vae("../model/rbm_cell_50",cell_arr)
print(drug_encodings)
print(cell_encodings)

def get_encoded_data(X,Y,drug_encodings,cell_encodings):
    X = X.detach().numpy()
    y =[]
    print(type(X))
    X_encoded = []
    for i in range(len(X)):
        embedding = []
        numpy_embedding = []
        cell_idx,drug_idx = int(X[i][0].item()),int(X[i][1].item())
        #print(cell_idx,drug_idx)
        embedding.extend(cell_encodings[cell_idx])
        embedding.extend(drug_encodings[drug_idx])
        for k in embedding:
            numpy_embedding.append(k.item())
        X_encoded.append(numpy_embedding)
    for i in range(len(Y)):
        y.append(Y[i].item())
    X_encoded = np.array(X_encoded)
    return X_encoded,y
x_train_encoded,y_train = get_encoded_data(X_train,Y_train,drug_encodings,cell_encodings)
x_valid_encoded, y_valid = get_encoded_data(X_valid,Y_valid,drug_encodings,cell_encodings)
print(x_train_encoded)
print(x_train_encoded.shape)
print(y_train)
print(len(y_train))
#print(x_valid_encoded)
print(x_valid_encoded.shape)
#print(y_valid)
print(len(y_valid))

from sklearn.svm import SVR

reg =  SVR(C=1.0,epsilon=0.2)

reg.fit(x_train_encoded,y_train)
y_pred =  reg.predict(x_train_encoded)

print(pearson_corr(torch.Tensor(y_train),torch.Tensor(y_pred)))

print("-----------------------------------SVR-----------------------------------")
from sklearn.svm import SVR

reg =  SVR(C=10,epsilon=0.2)

reg.fit(x_train_encoded,y_train)
y_pred_svr =  reg.predict(x_train_encoded)
print("-----6------")
from scipy import stats
svr_preas=stats.pearsonr(y_train, list(y_pred_svr))#pearson_corr(torch.Tensor(y_train),torch.Tensor(y_pred_svr))
print(svr_preas)
print("-----------------------------------ElasticNet-----------------------------------")
from sklearn.linear_model import ElasticNet

reg_en = ElasticNet(random_state=0)

reg_en.fit(x_train_encoded,y_train)
y_pred_en=reg_en.predict(x_valid_encoded)
en_preas=pearson_corr(torch.Tensor(y_valid),torch.Tensor(y_pred_en))
print(en_preas)
print("-----------------------------------RandomForest-----------------------------------")
'''from sklearn.linear_model import ElasticNet

reg_en = ElasticNet(random_state=0)

reg_en.fit(x_train_encoded,y_train)
y_pred_en=reg_en.predict(x_valid_encoded)
en_preas=pearson_corr(torch.Tensor(y_valid),torch.Tensor(y_pred_en))
print(en_preas)'''
from sklearn.ensemble import RandomForestRegressor

reg_rf = RandomForestRegressor(bootstrap=False, criterion='squared_error', max_depth=10,
           max_features=20, max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=-1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

reg_rf.fit(x_train_encoded,y_train)
y_pred_rf=reg_rf.predict(x_valid_encoded)
rf_preas=pearson_corr(torch.Tensor(y_valid),torch.Tensor(y_pred_rf))
print(rf_preas)

print("-----------------------------------LinearRegression-----------------------------------")
from sklearn.linear_model import LinearRegression

reg_lin =  LinearRegression()
reg_lin.fit(x_train_encoded,y_train)
y_pred_lin =  reg_lin.predict(x_valid_encoded)
print("-----6------")
lin_preas=pearson_corr(torch.Tensor(y_valid),torch.Tensor(y_pred_lin))
print(lin_preas)
print(type(y_train),type(y_pred_lin))

print("-----------------------------------MLP-----------------------------------")
from sklearn.neural_network import MLPRegressor
reg_mlp=MLPRegressor(random_state=1, max_iter=500,alpha=0.1,learning_rate='adaptive')
reg_mlp.fit(x_train_encoded,y_train)
y_pred_mlp =  reg_mlp.predict(x_valid_encoded)
print("-----6------")
mlp_preas=pearson_corr(torch.Tensor(y_valid),torch.Tensor(y_pred_mlp))
print(mlp_preas)
print(type(y_train),type(y_pred_lin))


print("---------------------------------------------------------------------------")


from scipy import stats
#-------------------Plots---------------
#--------------------------------------MLP--------------------------------------
xy_df_mlp=pd.DataFrame()
xy_df_mlp["Drug"]=(X_valid.T)[1]
xy_df_mlp["y_actual"]=y_valid
xy_df_mlp["y_pred"]=y_pred_mlp
unique_list=list(np.unique(xy_df_mlp['Drug']))
print(len(unique_list))
drugs=[]
pc=[]
for i in unique_list:
  new_df=xy_df_mlp[xy_df_mlp['Drug']==i]
  if len(new_df)>1:
      correlation, pvalue = stats.pearsonr(new_df['y_actual'], new_df['y_pred'])
      if not (math.isnan(correlation) or math.isnan(pvalue)):
          pc.append([correlation, pvalue])
          drugs.append(i)
  #pear_df['Drug']=i

pear_df=pd.DataFrame({"Drug":drugs,"pearson_correlation":np.array(pc).T[0],"pvalue":np.array(pc).T[1]})
'''print(np.mean(pear_df['pearson_correlation']))
print(pear_df)'''
pear_df.to_csv("mlp_pear.csv")


#--------------------------------------Linear Regression--------------------------------------
xy_df_lin=pd.DataFrame()
xy_df_lin["Drug"]=(X_valid.T)[1]
xy_df_lin["y_actual"]=y_valid
xy_df_lin["y_pred"]=y_pred_lin
unique_list_lin=list(np.unique(xy_df_lin['Drug']))
print(len(unique_list_lin))
drugs=[]
pc=[]
for i in unique_list_lin:
  new_df=xy_df_lin[xy_df_lin['Drug']==i]
  if len(new_df)>1:
      correlation, pvalue = stats.pearsonr(new_df['y_actual'], new_df['y_pred'])
      if not (math.isnan(correlation) or math.isnan(pvalue)):
          pc.append([correlation, pvalue])
          drugs.append(i)
  #pear_df['Drug']=i

pear_df_lin=pd.DataFrame({"Drug":drugs,"pearson_correlation":np.array(pc).T[0],"pvalue":np.array(pc).T[1]})
'''print(np.mean(pear_df['pearson_correlation']))
print(pear_df)'''
pear_df_lin.to_csv("lin_pear.csv")

#--------------------------------------SVR--------------------------------------
from sklearn.svm import SVR

reg =  SVR(C=10,epsilon=0.2)

reg.fit(x_train_encoded,y_train)
y_pred_svr =  reg.predict(x_valid_encoded)
print("-----6------")
from scipy import stats
svr_preas=stats.pearsonr(y_valid,(y_pred_svr))#pearson_corr(torch.Tensor(y_train),torch.Tensor(y_pred_svr))
print(svr_preas)
xy_df_svr=pd.DataFrame()
xy_df_svr["Drug"]=(X_valid.T)[1]
xy_df_svr["y_actual"]=y_valid
xy_df_svr["y_pred"]=y_pred_svr
unique_list_svr=list(np.unique(xy_df_svr['Drug']))
print(len(unique_list_svr))
drugs=[]
pc=[]
for i in unique_list_svr:
  new_df=xy_df_svr[xy_df_svr['Drug']==i]
  if len(new_df)>1:
      correlation, pvalue = stats.pearsonr(new_df['y_actual'], new_df['y_pred'])
      if not (math.isnan(correlation) or math.isnan(pvalue)):
          pc.append([correlation, pvalue])
          drugs.append(i)
  #pear_df['Drug']=i

pear_df_svr=pd.DataFrame({"Drug":drugs,"pearson_correlation":np.array(pc).T[0],"pvalue":np.array(pc).T[1]})
'''print(np.mean(pear_df['pearson_correlation']))
print(pear_df)'''
pear_df_svr.to_csv("lin_svr.csv")


#--------------------------------------Random Forest--------------------------------------
xy_df_rf=pd.DataFrame()
xy_df_rf["Drug"]=(X_valid.T)[1]
xy_df_rf["y_actual"]=y_valid
xy_df_rf["y_pred"]=y_pred_rf
unique_list_rf=list(np.unique(xy_df_rf['Drug']))
print(len(unique_list_rf))
drugs=[]
pc=[]
for i in unique_list_rf:
  new_df=xy_df_rf[xy_df_rf['Drug']==i]
  if len(new_df)>1:
      correlation, pvalue = stats.pearsonr(new_df['y_actual'], new_df['y_pred'])
      if not (math.isnan(correlation) or math.isnan(pvalue)):
          pc.append([correlation, pvalue])
          drugs.append(i)
  #pear_df['Drug']=i

pear_df_rf=pd.DataFrame({"Drug":drugs,"pearson_correlation":np.array(pc).T[0],"pvalue":np.array(pc).T[1]})
'''print(np.mean(pear_df['pearson_correlation']))
print(pear_df)'''
pear_df_rf.to_csv("lin_rf.csv")