import pandas as pd
import numpy as np
import statsmodels.api as sm

def forward_selection(X: pd.DataFrame, y: pd.Series) -> list:

    variables_selected = X[['intercept']]
    variables_remain = X.drop(columns=['intercept'])

    model = sm.OLS(y, variables_selected).fit()
    adj_r2_history = [model.rsquared_adj]

    iteration = 0

    print(f"Iteration {iteration}: start with {variables_selected.columns.tolist()} | Adj R2 = {model.rsquared_adj:.2%}")

    while True:
        iteration += 1

        variables_all_iteration = []
        adj_r2_all_iteration = []
        variable_names = []

        for variable in variables_remain:
            variables_1_iteration = variables_selected.join(variables_remain[[variable]])
            model_i = sm.OLS(y, variables_1_iteration).fit()

            variables_all_iteration.append(variables_1_iteration)
            adj_r2_all_iteration.append(model_i.rsquared_adj)
            variable_names.append(variable)

        if len(variables_remain.columns) == 0:
            print("No more variables to test → stopping")
            break

        best_r2 = max(adj_r2_all_iteration)
        best_idx = adj_r2_all_iteration.index(best_r2)
        best_variable = variable_names[best_idx]

        if best_r2 <= adj_r2_history[-1]:
            print("No improvement in Adjusted R² → stopping")
            break

        adj_r2_history.append(best_r2)
        variables_selected = variables_all_iteration[best_idx]
        variables_remain = variables_remain.drop(columns=[best_variable])

        print(f"Iteration {iteration}: + {best_variable} | Adj R2 = {best_r2:.2%} | Variables: {variables_selected.columns.tolist()}")

    if len(variables_remain.columns) > 0:
        print(f"Unused variables (no improvement in Adj R²): {variables_remain.columns.tolist()}")
    else:
        print("All variables were used")

    return variables_selected.columns