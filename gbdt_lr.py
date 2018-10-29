from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
from scipy.sparse.construct import hstack


data = pd.read_csv('./train.csv', sep='\t')

X = data[cols]
Y = data['y']
print('X shape:', X.shape)
dummies = pd.get_dummies(X['changyongdizhi'])
X = X.drop('changyongdizhi', axis=1)
X = X.join(dummies)
print('X join shape:', X.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

gbt_x, lr_x, gbt_y, lr_y = train_test_split(x_train, y_train, test_size=0.5)
gbt = GradientBoostingClassifier(n_estimators=30, min_samples_leaf=30, max_depth=10)

# model = gbt.fit(x_train, y_train)
# y_hat = gbt.predict(x_test)
# y_prob = gbt.predict_proba(x_test)
# print(accuracy_score(y_test, y_hat))
# print(roc_auc_score(y_test, y_prob[:, 1]))


gbt.fit(gbt_x, gbt_y)
gbt_features = gbt.apply(gbt_x)[:, :, 0]
print(gbt_features.shape)
enc_gbt = OneHotEncoder().fit(gbt_features)
lr_features = enc_gbt.transform(gbt.apply(lr_x)[:, :, 0])
lr_features = hstack([lr_features, lr_x])
clf = LogisticRegression(C=10, max_iter=10000)
clf.fit(lr_features, lr_y)
new_features = enc_gbt.transform(gbt.apply(x_test)[:, :, 0])
print('new', type(new_features), new_features.shape)
print('x_test', type(x_test), x_test.shape)

new_features = hstack([new_features, x_test])
print(new_features.shape)
y_hat = clf.predict(new_features)
y_prob = clf.predict_proba(new_features)

lr_features = enc_gbt.transform(gbt.apply(x_train)[:, :, 0])
lr_features = hstack([lr_features, x_train])
y_hat1 = clf.predict(lr_features)
print('train:',accuracy_score(y_hat1, y_train))
print(accuracy_score(y_test, y_hat))
print(roc_auc_score(y_test, y_prob[:, 1]))
print('-------------------------------------------')
