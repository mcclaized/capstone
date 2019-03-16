import os

import pandas as pd

DATA_DIR=os.getenv('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath((__file__)))), 'data'))

class SingleDayDataLoader(object):
    """
    This class loads one days worht of data from the vanguard data-set
    """

    PATH = os.path.join(DATA_DIR, 'vanguard_merge_sample.csv')

    def __init__(self, date=None):
        df = pd.read_csv(self.PATH)

        if date is None:
            date = max(df["date"].unique())

        self.data = (
            df
            .loc[df["date"] == date]
            .set_index("ISIN")
            .copy()
        )

    def get_bond(self, isin):
        return self.get_bonds([isin])

    def get_bonds(self, isins):
        return self.data.loc[isins]

    def get_cohort(self, bond, attributes):
        cohort_conditions = None
        for col in attributes:
            condition = self.data[col] == bond[col].values[0]
            if cohort_conditions is None:
                cohort_conditions = condition
            else:
                cohort_conditions = cohort_conditions & condition
        return self.data.loc[cohort_conditions]

