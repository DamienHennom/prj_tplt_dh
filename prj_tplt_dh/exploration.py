"""
Exploration of the data
"""

from pandas import Series, cut
import matplotlib.pyplot as plt
import seaborn as sns


class AverageResponse:
    """ Compute and plot average response by variable

    Parameters
    ----------
    data : pandas frame
        Data to use
    response : str, response variable in the data
        The response variable in the project
    bins : Number of bins for plotting, default=10
        Number of bins for plotting
    """

    def __init__(self, data, response, bins=10):
        self.data = data
        self.response = response
        self.bins = bins

    def compute_average_response(self, variable):
        """ Compute average response by variable

        Parameters
        ----------
        variable : str, variable to compute by
                The variable to compute by

        Returns
        -------
        recapTable : Recap table
            Returns Data recap
        """

        # Copy data
        data = self.data.copy()

        # Compute table
        if data[variable].dtype.name != "object" and len(data[variable].unique()) > self.bins:
            data["var_new"] = cut(data[variable], self.bins, duplicates='drop')
        else:
            data["var_new"] = data[variable].astype(str)

        # Final table
        def agg_func(x):
            names = {
                self.response: x[self.response].mean(),
                'count': x['var_new'].count()}

            return Series(names, index=[self.response, 'count'])

        recaptable = data.groupby("var_new").apply(agg_func).reset_index()
        recaptable.rename(columns={"var_new": variable}, inplace=True)

        return recaptable

    def plotting(self, recaptable):
        """ Plot recapTable

        Parameters
        ----------
        recaptable : pandas frame
                Created via compute_average_response

        Returns
        -------
        to_plot : matplotlib object
               Returns plot to be show
        """

        # Fill NA for plotting
        recaptable = recaptable.fillna(0).copy()

        # Plotting
        variable = recaptable.columns.values[0]
        f, ax = plt.subplots()
        ax2 = ax.twinx()
        sns.barplot(x=variable, y='count', data=recaptable, ax=ax, color="dodgerblue")
        sns.pointplot(x=variable, y=self.response, data=recaptable, ax=ax2, color="chartreuse")
        ax.set_xlabel(variable)
        ax.set_ylabel(variable)
        ax2.set_ylabel(self.response)
        plt.title("Average reponse by " + variable)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)

        return plt
