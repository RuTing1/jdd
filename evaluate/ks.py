import pandas as pa
import numpy as np


def pct_rank_qcut(series, n):
    edges = pa.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1, ascending=True).apply(f)


def KS(df, dep, score):
    df['score_rank'] = pct_rank_qcut(df[score], 10)

    #KS Analysis on dev sample

    total = df[dep].count()
    total_responder = df[dep].sum()
    total_non_responder = total-total_responder

    total_int = df.groupby(['score_rank'])['score_rank'].count()
    Responder = df.groupby(['score_rank'])[dep].sum()
    NonResponder = total_int - Responder
    Min_Score = df.groupby(['score_rank'])[score].min()
    Mean_Score = df.groupby(['score_rank'])[score].mean()
    Max_Score = df.groupby(['score_rank'])[score].max()

    PercResponder = (Responder/total_responder)*100
    CumPercResponder = (np.cumsum(Responder)/total_responder)*100
    CumPercNonResponder = (np.cumsum(NonResponder)/total_non_responder)*100
    ResponseRate = (Responder/total_int)*100
    KS = CumPercResponder-CumPercNonResponder
    
    dev_name = [total_int,Min_Score,Mean_Score,Max_Score,Responder,PercResponder,CumPercResponder,ResponseRate,KS]
    columns = ["Total_Int","Min_Score","Mean_Score","Max_Score","Responder","PercResponder","CumPercResponder","ResponseRate","KS"]
    KS_dev = pa.concat(dev_name,keys=columns,axis=1)
    S = pa.Series([total,df[score].min(),df[score].mean(),df[score].max(),
                 total_responder, 100, "", total_responder/total, abs(KS).max()], index=columns)
    KS_dev = KS_dev.append(S,ignore_index=True)
    return KS_dev


if __name__ == '__main__':
    
    dev_pred_pangu_score = pa.read_csv('D:/ZRWORK/ks_auc/ppd1215/union_pay_score.csv')
    dev_pred_result_1 = KS(dev_pred_pangu_score, "dep", "score")
    print(dev_pred_result_1)




