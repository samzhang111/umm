from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

import pandas as pd
import sys
import pickle
import pprint

rfc = RandomForestClassifier()

# Read training file from standard in
df = pd.read_csv(sys.argv[1])

X = df.ix[:,:-1].values
Y = df.ix[:,-1].values

kf = StratifiedKFold(y=Y, n_folds=5, shuffle=True, random_state=983214)

fprs = []
tprs = []
thresholds = []
for train, test in kf:
    rfc.fit(X[train], Y[train])
    predictions = rfc.predict_proba(X[test])
    #pprint.pprint( zip(Y[test], predictions) )
    fpr, tpr, threshold = roc_curve(Y[test], predictions[:, 1])
    fprs.append(fpr)
    tprs.append(tpr)
    thresholds.append(threshold)

for i in range(len(fprs)):
    plt.plot(fprs[i], tprs[i], lw=1)

plt.show()

rfc.fit(X, Y)

with open('model.p', 'wb') as p:
    pickle.dump(rfc, p)
