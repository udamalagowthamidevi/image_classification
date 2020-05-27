import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import time
start_time = time.time()



train_df = pd.read_csv('train.csv')

X_tr = train_df.values[:, 1:].astype(float)
Y_tr = train_df.values[:, 0]

print ('training...')
clf = RandomForestClassifier(100)
clf = clf.fit(X_tr, Y_tr)
print ('training complete...')

scores = cross_val_score(clf, X_tr, Y_tr)
print ('Accuracy {0}'.format(np.mean(scores)))


# Read test datatest_df = pd.read_csv('test.csv')
X_test = test_df.values.astype(float)

# make predictionsY_test = clf.predict(X_test)
# make DF to print easilyans = pd.DataFrame(data={'ImageId':range(1,len(Y_test)+1), 'Label':Y_test})

# save to csvans.to_csv('rf.csv', index=False)

print("--- %s seconds ---" % (time.time() - start_time))
