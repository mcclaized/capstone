import os
import pandas as pd
import datetime

DATA_DIR=os.getenv(
    'DATA_DIR', 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath((__file__)))), 'data')
)

class SingleDayDataLoader(object):
    """
    This class loads one days worth of data from the vanguard data-set
    """

    PATH = os.path.join(DATA_DIR, 'vanguard_merge.csv')

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

class MultipleDayDataLoader(object):
    """
    This class loads one days worth of data from the vanguard data-set
    """

    PATH = os.path.join(DATA_DIR, 'vanguard_merge.csv')

    def __init__(self, date=None, numdays=None):
        df = pd.read_csv(self.PATH)

        if date is None:
            date = max(df["date"].unique())

        if numdays is None:
            numdays = 30

        date_list = [pd.to_datetime(date) - datetime.timedelta(days=x) for x in range(0, numdays)]

        date_list_str = list(map(lambda x: x.strftime('%Y-%m-%d'), date_list))

        self.data = (
            df
            .loc[df["date"].isin(date_list_str)]
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