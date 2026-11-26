"""Reproduce all temperature experiments from the repository root."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from temperature_analysis import run_analysis as a

OUT = a.OUT

def predict(name, alpha, x, y, z):
    if name in ['Random Forest','Gradient Boosting','SVR','KNN']:
        model = {'Random Forest': lambda: RandomForestRegressor(n_estimators=100,max_depth=int(alpha),random_state=42), 'Gradient Boosting':lambda: GradientBoostingRegressor(n_estimators=100,max_depth=int(alpha),random_state=42), 'SVR':lambda:make_pipeline(StandardScaler(),SVR(C=alpha)), 'KNN':lambda:make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=int(alpha)))}[name]()
        return model.fit(np.asarray(x).reshape(-1,1),y).predict(np.asarray(z).reshape(-1,1))
    if name == 'Training mean': return np.repeat(np.mean(y),len(z))
    return a.fit_predict(name,alpha,x,y,z)

def main():
    a.main()
    df=a.load_data(); train=df[df.YEAR<=2001]; test=df[df.YEAR>=2002]
    candidates=a.candidates()+[(n,v) for n,vs in [('Random Forest',[2,5]),('Gradient Boosting',[1,2]),('SVR',[1,10]),('KNN',[3,5])] for v in vs]+[('Training mean',None)]
    cv=[]; results=[]; predictions=[]
    for target in a.TARGETS:
        scores=[]
        for name,param in candidates:
            errors=[]
            for start,end in a.FOLDS:
                tr=train[train.YEAR<start]; va=train[train.YEAR.between(start,end)]
                err=a.rmse(va[target],predict(name,param,tr.YEAR,tr[target],va.YEAR)); errors.append(err)
                cv.append([target,name,param,start,end,err])
            scores.append((name,param,float(np.mean(errors))))
        best=min((s for s in scores if s[0]!='Training mean'),key=lambda s:s[2])
        for name in dict.fromkeys(s[0] for s in scores):
            n,p,c=min((s for s in scores if s[0]==name),key=lambda s:s[2]); pred=predict(n,p,train.YEAR,train[target],test.YEAR)
            results.append([target,n,p,c,mean_absolute_error(test[target],pred),a.rmse(test[target],pred),r2_score(test[target],pred),(n,p)==best[:2]])
            if (n,p)==best[:2]:
                predictions.extend(zip([target]*len(test),test.YEAR,test[target],pred))
    pd.DataFrame(cv,columns=['Series','Model','Parameter','Validation_start','Validation_end','RMSE']).to_csv(a.TABLES/'extended_cv.csv',index=False)
    r=pd.DataFrame(results,columns=['Series','Model','Parameter','CV_RMSE','MAE','RMSE','R2','Selected']); r.to_csv(a.TABLES/'extended_holdout.csv',index=False)
    p=pd.DataFrame(predictions,columns=['Series','Year','Actual','Predicted']); p.to_csv(a.TABLES/'extended_predictions.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,6)); r.pivot(index='Model',columns='Series',values='RMSE').plot.bar(ax=ax); ax.set_ylabel('Holdout RMSE (°C)'); fig.tight_layout(); fig.savefig(a.FIGURES/'model_comparison.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(); q=p[p.Series=='ANNUAL']; ax.plot(q.Year,q.Actual,label='Actual'); ax.plot(q.Year,q.Predicted,label='Predicted'); ax.legend(); ax.set_ylabel('Temperature (°C)'); fig.tight_layout(); fig.savefig(a.FIGURES/'annual_selected_prediction.png',dpi=150); plt.close(fig)
    Path('results.txt').write_text(r.to_string(index=False)+'\n',encoding='utf-8')
if __name__=='__main__': main()
