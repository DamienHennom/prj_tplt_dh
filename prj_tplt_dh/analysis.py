"""
Sklearn model analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import inf


class ModelMarginal:
    """ Compute and plot marginal effects for model

    Parameters
    ----------
    model : sklearn model
        Model to work with
    X : pandas frame
        data used to compute marginal
    y : list
        actual response for X
    bins : Number of bins for plotting, default=20
        Number of bins for plotting
    """

    def __init__(self, model, X, y, bins=20):
        self.model = model
        self.X = X
        self.y = y
        self.bins = bins

    def compute_marginal(self, var):
        """ Compute marginal effect for variable var

        Parameters
        ----------
        var : str, variable to compute by
                The variable to compute by

        Returns
        -------
        marginals : pandas frame
            Marginal effects
        """
        # Actual
        actual = self.y

        # Prediction
        pred = self.model.predict(self.X)

        # Unique values computation
        if len(self.X[var].unique()) > self.bins:
            unique_value = pd.cut(self.X[var], self.bins, duplicates='drop')
            unique_value = list(set([(a.left + a.right) / 2 for a in unique_value]))
        else:
            unique_value = list(self.X[var].unique())

        unique_value = sorted(unique_value)

        # Compute marginal
        Xused = self.X.copy()
        value_before = -inf
        marginals = pd.DataFrame([])

        for value in unique_value:
            # Compute actual
            mean_value_actual = np.nanmean(list(actual[(pd.Series(self.X[var]) <= value) &
                                                       (pd.Series(self.X[var]) > value_before)]))
            # Compute predicted
            mean_value_pred = np.nanmean(list(pred[(pd.Series(self.X[var]) <= value) &
                                                   (pd.Series(self.X[var]) > value_before)]))
            # Count values
            count_value = len(list(pred[(pd.Series(self.X[var]) <= value) &
                                        (pd.Series(self.X[var]) > value_before)]))
            # Change value
            Xused[var] = value
            # Compute marginal effect
            pred_marginal_mean = np.mean(self.model.predict(Xused))

            # Store results
            result_frame = pd.DataFrame({var: [value], 'Marginal': [pred_marginal_mean], 'Actual': [mean_value_actual],
                                         'Pred': [mean_value_pred], 'Count': [count_value]})
            marginals = marginals.append(result_frame)

            # Update value
            value_before = value

        return marginals

    def plotting(self, marginals):
        """ Plot marginals

        Parameters
        ----------
        marginals : pandas frame
                Created via compute_marginal

        Returns
        -------
        plt : matplotlib object
               Returns plot to be show
        """
        # Find var
        var = marginals.columns.values[0]

        # Plot
        f, ax = plt.subplots()
        ax2 = ax.twinx()
        sns.barplot(x=var, y='Count', data=marginals, ax=ax, color="dodgerblue")
        sns.pointplot(x=var, y='Actual', data=marginals, ax=ax2, color="chartreuse", label="Actual")
        sns.pointplot(x=var, y='Pred', data=marginals, ax=ax2, color="orange", label="Prediction")
        sns.pointplot(x=var, y='Marginal', data=marginals, ax=ax2, color="black", label="Marginal")
        ax.set_xlabel(var)
        ax.set_ylabel('Count')
        ax2.set_ylabel('Average')
        ax2.legend(handles=ax2.lines[::len(marginals) + 1], labels=["Actual", "Prediction", "Marginal"])
        plt.title("Marginal effect " + var)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)

        return plt
