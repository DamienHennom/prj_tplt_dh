"""
Data preparation
"""

import pandas as pd


class FillData:
    """ Compute and plot average response by variable

    Parameters
    ----------
    data : pandas frame
        Data to use
    numeric_type : list
        List of numeric types
    """

    def __init__(self, data):
        self.data = data
        self.numeric_type = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    def fill_cat(self):
        """ Fill categorical part of the data with "Miss"

        Parameters
        ----------

        Returns
        -------
        data_cat : pandas frame
            Returns categorical part of the data filled
        """

        data_cat = self.data.select_dtypes(exclude=self.numeric_type)
        data_cat = data_cat.fillna("Miss")
        return data_cat

    def fill_num(self):
        """ Fill numerical part of the data with -9999

        Parameters
        ----------

        Returns
        -------
        data_num : pandas frame
            Returns numerical part of the data filled
        """
        data_num = self.data.select_dtypes(include=self.numeric_type)
        data_num = data_num.fillna(-9999)

        return data_num

    def fill_all(self):
        """ Fill categorical and numerical part of the data

        Parameters
        ----------

        Returns
        -------
        data_filled : pandas frame
            Returns data filled
        """
        data_cat = self.fill_cat()
        data_num = self.fill_num()

        # Join data
        data_filled = pd.concat([data_cat, data_num], axis=1, sort=False)

        return data_filled