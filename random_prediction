import pandas as pd
import numpy as np

orig = pd.read_csv("/Volumes/DataScience/stage1_sample_submission.csv")
ID = pd.DataFrame(orig['id'])
pred = pd.DataFrame(np.random.rand(198,1), dtype='float32')
pred.columns = ['cancer']
pred_032917 = pd.concat([ID,pred], axis=1)
pred_032917.to_csv('pred_032917.csv', index=False)
