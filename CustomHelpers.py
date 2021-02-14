import pandas as pd
from sklearn.feature_selection import chi2, f_classif
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_complement(sup:pd.DataFrame, sub:pd.DataFrame) -> pd.DataFrame:
    """
    Gets the rows in `sup` which aren't in `sub`.
    Assumes that `sub` is a subset of `sup` and the indices align.
    """
    sub_idx = list(sub.index)
    sup_idx = list(sup.index)
    complement_idx = []
    for i in sub_idx:
        sup_idx.remove(i)
        
    complement = sup.iloc[complement_idx, :]
    return complement


def stacked_hist(feature:str, df:pd.DataFrame, target:str):
    """
    Returns a stacked histogram where the feature values in the supplied DataFrame
    are grouped by color based on their target values. 
    """
    plt.hist(x=[df.loc[df[target] == True,  feature],
                df.loc[df[target] == False, feature]],
             bins=200, stacked=True, color=['tab:orange','tab:blue'])
    plt.xlabel(feature)
    
    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['tab:orange','tab:blue']]
    labels= [f'{target}: True', f'{target}: False']
    plt.legend(handles, labels)

    plt.show()


def get_chi2(X, y):
    """
    Returns a dictionary where the keys are the features and the values are tuples (chi2, p-value).
    """
    X_dummy = pd.get_dummies(X)
    stats = {}
    features = X.columns
    stats_list = chi2(X_dummy, y)

    for i in range(len(features)):
        stats[features[i]] = (stats_list[0][i], stats_list[1][i])

    return stats


def get_anova(X, y):
    """
    Returns a dictionary where the keys are the features and the values are tuples (chi2, p-value).
    """
    X_dummy = pd.get_dummies(X)
    stats = {}
    features = X.columns
    stats_list = f_classif(X_dummy, y)

    for i in range(len(features)):
        stats[features[i]] = (stats_list[0][i], stats_list[1][i])

    return stats


def univariate_summary(feature:str, df:pd.DataFrame, target:str, 
                       chi2_stats:dict, anova_stats:dict, alpha:float=0.05):
    """
    Returns some exploratory information about a single column in the supplied DataFrame.
    Use `get_chi2` and `get_anova` to supply the `chi2_stats` and `anova_stats` arguments.
    """
    stacked_hist(feature, df=df, target='Attrition_Flag')

    try:
        print(f'Mean: {df[feature].mean()}')
        print(f'Standard Deviation: {df[feature].std()}')
        print(f'Minimum: {df[feature].min()}')
        print(f'Maximum: {df[feature].max()}\n')
    except:
        pass

    chi2_p = chi2_stats[feature][1]
    chi2_is_ind = chi2_p < alpha

    anova_p = anova_stats[feature][1]
    anova_is_ind = anova_p < alpha

    print(f'Chi^2 p-value: {chi2_p:.3f}')
    print(f'Chi^2: {feature} is probably independent of target: {chi2_is_ind}\n')

    print(f'ANOVA F-Test p-value: {anova_p:.3f}')
    print(f'ANOVA: {feature} is probably independent of target: {anova_is_ind}')


def get_sorted_correlations(df:pd.DataFrame) -> pd.DataFrame:
    """
    Inputs a pandas DataFrame and returns a Pandas DataFrame with rows indexed 
    by pairs (feature1, feature2). The absolute value of the correlation between
    each pair is returned as a column and the rows are sorted by this column in 
    descending order.
    """
    features1 = list(df.corr().columns.copy())
    features2 = list(df.corr().index.copy())
    n = len(features1)
    num_rows = int(n*(n-1)/2)
    rows = [i for i in range(num_rows)]

    corrs_df = df.corr()
    pairs = []
    for f in features1:
        features2.remove(f)
        for g in features2:
            pairs.append( ((f, g), corrs_df.loc[f, g]) )

    corrs_sorted = pd.DataFrame(index=rows, columns=('Features', 'Abs_Correlation'))
    for row in rows:
        pair = pairs.pop(0)
        corrs_sorted.loc[row, 'Features'] = pair[0]
        corrs_sorted.loc[row, 'Abs_Correlation'] = pair[1]

    corrs_sorted['Is_Positive_Correlation'] = corrs_sorted['Abs_Correlation'] > 0
    corrs_sorted['Abs_Correlation'] = corrs_sorted['Abs_Correlation'].abs()
    corrs_sorted.sort_values(by='Abs_Correlation', ascending=False, inplace=True)
    corrs_sorted.reset_index(drop=True, inplace=True)    

    return corrs_sorted 


def stacked_scatterplot(df:pd.DataFrame, feature1:str, feature2:str, target:str):
    """
    Returns a stacked scatterplot where the feature values in the supplied DataFrame
    are grouped by color based on their target values. 
    """
    p0 = df.loc[df[target] == 0, [feature1, feature2]]
    p1 = df.loc[df[target] == 1, [feature1, feature2]]

    x0 = p0[feature1]
    y0 = p0[feature2]

    x1 = p1[feature1]
    y1 = p1[feature2]

    plt.scatter(x0, y0, c='tab:blue', label=f'{target}: False')
    plt.scatter(x1, y1, c='tab:orange', label=f'{target}: True')

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.show()


def compare_pr_curves(X, y, models:dict):
    """
    Plots precision vs recall for the supplied models, given as a dict of name, model pairs.
    """

    for model_name in models.keys():
        model = models[model_name]
        try:
            y_scores = model.predict_proba(X)[:,1]
            precisions, recalls, _ = precision_recall_curve(y, y_scores)
            plt.plot(recalls, precisions, label=model_name)
        except:
            print(f'Warning: Could not get scores from {model_name}.\nModel may not be fit.')
    
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve Comparison')
    plt.legend(loc='lower left')
    plt.show()


def show_precision_recall_vs_threshold(X, y, model, title='Precision/Recall vs Thresholds'):
    y_scores = model.predict_proba(X)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.legend()
    plt.grid()
    plt.show()

def predict_with_threshold(X, model, threshold:float):
    y_hat = model.predict_proba(X)[:,1]
    return y_hat >= threshold