import pandas as pd
import numpy as np
import statsmodels.api as sm

def backward_selection(X: pd.DataFrame, y: pd.Series, alpha: float = 0.05) -> list:

    variables_selected = X.copy()

    model = sm.OLS(y, variables_selected).fit()
    iteration = 0

    print(f"Iteration {iteration}: start with {variables_selected.columns.tolist()}")

    while True:
        iteration += 1

        pvalues_list = model.pvalues

        if 'intercept' in pvalues_list.index:
            pvalues_list = pvalues_list.drop('intercept')

        if len(pvalues_list) == 0:
            print("No variables to test → stopping")
            break

        worst_pvalue = pvalues_list.max()
        worst_variable = pvalues_list.idxmax()

        if worst_pvalue <= alpha:
            print("All variables significant → stopping")
            break

        variables_selected = variables_selected.drop(columns=[worst_variable])
        model = sm.OLS(y, variables_selected).fit()

        print(f"Iteration {iteration}: - {worst_variable} | p-value = {worst_pvalue:.4f} | Variables: {variables_selected.columns.tolist()}")

    return variables_selected.columns