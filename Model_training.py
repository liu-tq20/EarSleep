import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
# from sklearn.svm import SVC
from features_acc import extract_features_acc # make sure features.py is in the same directory
from features_audio import extract_features_audio
from util import slidingWindow, reorient, reset_vars
# from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import datetime


print("Loading data...")
sys.stdout.flush()


# print(absolute_path)  # check path
# if os.path.exists(absolute_path):
#     print("File found!")
# else:
#     print("File not found!")

data_path =  './data'

print("Started loading...")
data_file_bp_08 = os.path.join(data_path, 'accel-08.csv')
data_bp_08 = np.loadtxt(data_file_bp_08, delimiter=',', dtype = object, converters = {0: np.float, 1: np.float, 2: np.float, 3: lambda t: datetime.strptime(t.decode("utf-8"), "%H:%M")})
data_bp_08 = np.insert(data_bp_08, 3, 0, axis = 1)
hdata_file_bp_08 = os.path.join(data_path, 'BPM-08.csv')
hdata_bp_08 = np.loadtxt(hdata_file_bp_08, delimiter=',', dtype = object, converters = {0: lambda t: datetime.strptime(t.decode("utf-8"), "%H:%M"), 1: np.float, 2: np.int})


data_file_sc_09 = os.path.join(data_path, 'accel-09.csv')
data_sc_09 = np.loadtxt(data_file_sc_09, delimiter=',', dtype = object, converters = {0: np.float, 1: np.float, 2: np.float, 3: lambda t: datetime.strptime(t.decode("utf-8"), "%H:%M:%S")})
data_sc_09 = np.insert(data_sc_09, 3, 0, axis = 1)
hdata_file_sc_09 = os.path.join(data_path, 'BPM-09.csv')
hdata_sc_09 = np.loadtxt(hdata_file_sc_09, delimiter=',', dtype = object, converters = {0: lambda t: datetime.strptime(t.decode("utf-8"), "%H:%M"), 1: np.float, 2: np.int})


data_file_aa_11 = os.path.join(data_path, 'accel-11.csv')
data_aa_11 = np.loadtxt(data_file_aa_11, delimiter=',', dtype = object, converters = {0: np.float, 1: np.float, 2: np.float, 3: lambda t: datetime.strptime(t.decode("utf-8"), "%H:%M")})
data_aa_11 = np.insert(data_aa_11, 3, 0, axis = 1)
hdata_file_aa_11 = os.path.join(data_path, 'BPM-11.csv')
hdata_aa_11 = np.loadtxt(hdata_file_aa_11, delimiter=',', dtype = object, converters = {0: lambda t: datetime.strptime(t.decode("utf-8"), "%H:%M"), 1: np.float, 2: np.int})


data_file_nm_18 = os.path.join(data_path, 'accel-18.csv')
data_nm_18 = np.loadtxt(data_file_nm_18, delimiter=',', dtype = object, converters = {0: np.float, 1: np.float, 2: np.float, 3: lambda t: datetime.strptime(t.decode("utf-8"), "%H:%M:%S")})
data_nm_18 = np.insert(data_nm_18, 3, 0, axis = 1)
hdata_file_nm_18 = os.path.join(data_path, 'BPM-18.csv')
hdata_nm_18 = np.loadtxt(hdata_file_nm_18, delimiter=',', dtype = object, converters = {0: lambda t: datetime.strptime(t.decode("utf-8"), "%H:%M"), 1: np.float, 2: np.int})

data = np.vstack((data_sc_09, data_nm_18))
data = np.vstack((data, data_bp_08))
data = np.vstack((data, data_aa_11))

hdata = np.vstack((hdata_sc_09, hdata_nm_18))
hdata = np.vstack((hdata, hdata_bp_08))
hdata = np.vstack((hdata, hdata_aa_11))

print("All data loaded.")

sys.stdout.flush()


# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels from Acc
#
# -----------------------------------------------------------------------------

# you may want to play around with the window and step sizes
window_size = 20
step_size = 20


# step_size
# sampling rate for the sample data should be about 25 Hz; take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

feature_names = ["Mean", "Variance", "Local Mimimum Count", "Local Maximum Count", "Range", "Magnitude of Dominant Frequency", "Mean Heart Rate"]
class_names = ["Awake","Light Sleep","Deep Sleep"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

n_features = len(feature_names)

X = np.zeros((0,n_features))
y = np.zeros(0,)

#because hr data in backwards
count = len(hdata)-1
print("hdata: ")
print(hdata)

print("data: ")
print(data)
for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    temp = np.zeros((1,3))
     #while time at row count is under time at accel, increase count (move to next row)
     #only have one window. Each row in window has own observation that needs hr
    for row in range(len(window_with_timestamp_and_label)):
        while hdata[count][0] < window_with_timestamp_and_label[row][4] and count > 0:
            count=count-1
            print("changed count ", count)
        #remove timestamps from accel data
        temp = np.vstack((temp,window_with_timestamp_and_label[row][:-2]))
        #add hr data to accel
        hr_label = np.append(hdata[count][1], hdata[count][2])
        window_with_timestamp_and_label[row] = np.append(temp[row+1], hr_label)
        #add in label (hr_data is on form hr, t, label)
        #remove time and label for feature extraction
    window = window_with_timestamp_and_label[:,:-1]
    # extract features over window:
    x = extract_features_acc(window) #x, y, z, t (not reoriented)  -> x, y, z, heart rate, label/class -> x, y, z, hr
    # append features:
    # shapes into 1 row with unspecified number of columns (so just 1 row of n_features)
    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    # append label:
    y = np.append(y, window_with_timestamp_and_label[10, -1]) #we don't know why this is 10?


print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels from Microphone
#
# -----------------------------------------------------------------------------


# %%---------------------------------------------------------------------------
#
#		                Debug
#
# -----------------------------------------------------------------------------

print("X===================")
print(X)
print("y===================")
print(y)
print("X.shape===================")
print(X.shape)
print("y.shape===================")
print(y.shape)

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

# Report average accuracy, precision and recall metrics.

clf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=10)
# clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, max_features = n_features)
# clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, max_features = 5)
# clf=SVC()

# cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
cv = KFold(n_splits=10, shuffle=True, random_state=None)

def compute_accuracy_all(conf):
    r0c0 = conf[0][0]
    r1c1 = conf[1][1]
    r2c2 = conf[2][2]
    accuracy = float(r0c0+r1c1+r2c2)/np.sum(conf)
    print("accuracy: {}".format(accuracy))
    return accuracy

def compute_accuracy(conf, col):
    #TP + TN/(P + N)
    rc = conf[col][col]
    accuracy = float(rc)/np.sum(conf[0][col]+conf[1][col]+conf[2][col])
    print("accuracy {}: {}".format(col, accuracy))
    return accuracy

def compute_recall(conf, col):
    #actual = column, predicted = row
    #TP/(TP+FN), col-wise
    #col = 0,1,2
    row_tp = col
    if col == 0:
        row2 = col+1
        row3 = col+2
    if col == 1:
        row2 = col-1
        row3 = col+1
    if col == 2:
        row2 = col-2
        row3 = col-1


    TP = float(conf[row_tp][col])
    FN = float(conf[row2][col])+float(conf[row3][col])
    recall = (TP)/(TP + FN) if (TP+FN !=0) else -5
    #print("recall ",conf[col_t][row], conf[col2][row], conf[col3][row])
    print("recall {}: {}".format(col, recall))
    return recall

def compute_precision(conf, row):
    #TP/(TP+FP), row-wise
    col_tp = row
    if row == 0:
        col2 = row+1
        col3 = row+2
    if row == 1:
        col2 = row-1
        col3 = row+1
    if row == 2:
        col2 = row-2
        col3 = row-1


    TP = float(conf[row][col_tp])
    FP = float(conf[row][col2])+float(conf[row][col3])
    precision = (TP)/(TP + FP) if (TP+FP !=0) else -5
    print("precision {}: {}".format(row, precision))
    return precision

fold = np.zeros([10,10])
#rows:
#acc
#pre 1
#pre 2
#pre 3
#rec 1
#rec 2
#rec 3

for i, (train_indexes, test_indexes) in enumerate(cv.split(X)):
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]
    clf.fit(X_train, y_train)

    # predict the labels on the test data
    y_pred = clf.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    # deep, light, awake
    conf = confusion_matrix(y_test, y_pred, labels=[0,1,2])
    
    print("Fold {} : The confusion matrix is :".format(i))
    print(conf)
    acc = compute_accuracy_all(conf)
    fold[0,i] = acc
    for j in range(3):
        acc = compute_accuracy(conf, j)
        pre = compute_precision(conf, j)
        rec = compute_recall(conf,j)
        fold[1+j, i] = acc
        fold[4+j, i] = pre
        fold[7+j, i] = rec

    print("\n")

print("fold: ", fold)
avg_conf = np.mean(fold, axis = 1)
print("avg_conf: ", avg_conf)

print("average accuracy: {0:.3f}".format(avg_conf[0]))
print("average accuracy awake: {0:.3f}".format(avg_conf[1]))
print("average precision awake: {0:.3f}".format(avg_conf[2]))
print("average recall awake: {0:.3f}".format(avg_conf[3]))
print("average accuracy light sleep: {0:.3f}".format(avg_conf[4]))
print("average precision light sleep: {0:.3f}".format(avg_conf[5]))
print("average recall light sleep: {0:.3f}".format(avg_conf[6]))
print("average accuracy deep sleep: {0:.3f}".format(avg_conf[7]))
print("average precision deep sleep: {0:.3f}".format(avg_conf[8]))
print("average recall deep sleep: {0:.3f}".format(avg_conf[9]))
    

# when ready, set this to the best model you found, trained on all the data:
best_classifier = clf
with open('classifier.pickle', 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f, protocol=pickle.HIGHEST_PROTOCOL)