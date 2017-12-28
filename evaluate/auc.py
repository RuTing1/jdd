import pandas as pa
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def predictive_Accu(df, dep, score):
    
    fpr_dev = dict()
    tpr_dev = dict()
    roc_auc_dev = dict() 
    fpr_dev, tpr_dev, _ = roc_curve(df[dep], -df[score])
    roc_auc_dev = auc(fpr_dev, tpr_dev)

    dev_roc = {"fpr_dev":fpr_dev,"tpr_dev":tpr_dev}
    Dev_Roc = pa.DataFrame(dev_roc, columns=["fpr_dev", "tpr_dev"])
 
    return Dev_Roc, roc_auc_dev


# dev_pred = pa.read_csv('D:/ZRWORK/ks_auc/ppd1110/pangu_score_ks_auc.csv')
# dev_pred = pa.read_csv('D:/ZRWORK/ks_auc/ppd1110/pangu_score_old_ks_auc.csv')
# dev_pred = pa.read_csv('D:/ZRWORK/ks_auc/ppd1110/union_pay_score_ks_auc.csv')
# dev_pred = pa.read_csv('D:/ZRWORK/ks_auc/ppd1110/hulu_score_ks_auc.csv')
# dev_pred = pa.read_csv('D:/ZRWORK/ks_auc/ppd1110/ascore_ks_auc.csv')
# dev_pred = pa.read_csv('D:/ZRWORK/ks_auc/ppd1110/bscore_ks_auc.csv')
# dev_pred = pa.read_csv('D:/ZRWORK/ks_auc/ppd1110/ascore_old_ks_auc.csv')
dev_pred = pa.read_csv('D:/ZRWORK/ks_auc/ppd1215/pangu_score.csv')


dev_roc, dev_auc = predictive_Accu(dev_pred, "dep", "score")

# Plot of a ROC curve for a specific class

plt.figure()
lw = 2
plt.plot(dev_roc['fpr_dev'], dev_roc['tpr_dev'], color='darkorange', lw=lw,  label='ROC curve(Dev)(area = %0.3f)' % dev_auc)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
