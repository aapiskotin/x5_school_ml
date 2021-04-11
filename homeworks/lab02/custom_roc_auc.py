import pandas as pd
import numpy as np


def custom_roc_curve(y_true, y_pred):
    # Get table sorted by probabilities
    y_df = pd.DataFrame({'true': y_true,
                         'pred': y_pred})
    y_df.sort_values('pred', ascending=False, inplace=True)
    # Calculate TPR and FPR
    y_df['is_p'] = y_df.true == 1
    y_df['tpr'] = (y_df.is_p.cumsum() - 1) / (y_df.is_p.sum() - 1)
    y_df['fpr'] = (~y_df.is_p).cumsum() / (~y_df.is_p).sum()
    # Get correction for the case when probabilities are equal
    y_df = y_df.groupby('pred').first().reset_index()
    y_df = y_df.sort_values('pred', ascending=False)
    y_df = y_df.append(y_df.iloc[-1])
    y_df.iloc[-1, -1] = 1

    fpr, tpr = np.array(y_df.fpr), np.array(y_df.tpr)
    return fpr, tpr


def custom_auc(fpr, tpr):
    # Calculate as simple sum of areas of trapeze
    x_delta = fpr[1:] - fpr[:-1]
    auc = np.sum((tpr[1:] + tpr[:-1])/2 * x_delta)
    return auc
