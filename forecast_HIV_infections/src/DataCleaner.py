# Modify path so we can import from src if needed.
import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import pandas as pd

class DataCleaner:
    """Cleans the HIV data.

    Example usage:
        >>> from src.DataCleaner import DataCleaner
        >>> df = pd.read_csv('data/main.csv')
        >>> cleaner = DataCleaner()
        >>> df = cleaner.clean(df)
    
    """

    def __init__(self):
        pass
    
    def clean(self, df):
        df = df.copy()

        # Drop any row with a null
        df = df.dropna()

        # Drop the HIVincidence outlier of ~750.
        df = df[df['HIVincidence']<750]

        self.df = df

        return self.df

if __name__ == "__main__":
    # Example us
