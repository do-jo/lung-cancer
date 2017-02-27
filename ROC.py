
# coding: utf-8

# This function does below things:
# 1. Calculates [fpr, tpr, thresholds, auc] and matches the list to its key value (trial set #) in the dictionary named PRI (meaning 'Positive Rate Information').
# 2. Shows the ROC plot of each trial sets.
# 
# The input data should be in a 3-D numpy ndarray :
# (# of trials sets, # of training set, 2)
# eg) data.shape = (5, 300, 2) would mean 5 different trial sets, 300 is for the number of training sets in each trial, and 2 is for y value, and predicted value in the right order.
# 

# In[ ]:

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[ ]:

colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange']
# The number of Colors may vary depending on the number of trials sets
# eg. if we have 5 sets to check, have 5 different colors in the list!


# In[ ]:

PRI = {} # Postive Rate Information, keys = Trial set, values = list of fpr, tpr, thresholds, and roc_auc


# In[ ]:

def ROC(my_data):
    nb = my_data.shape[0] # nb indicates the number of trial sets in my_data 
    i = 1
    for k, c in zip(range(nb), colors):
    # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(my_data[k,:,0], my_data[k,:,1]) # [k,:,0] = y, [k,:,1] = pred
        roc_auc = auc(fpr, tpr)  # calculates the area under roc curve
        PRI[k] = [fpr, tpr, thresholds, roc_auc]
        plt.plot(fpr, tpr, lw=2, color=c,label='ROC fold %d (area = %0.2f)' %(i, roc_auc)) 
        # the lable indicates the matching trial set & its area under ROC curve to the 2decimal points
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:



