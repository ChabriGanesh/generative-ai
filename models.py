import pandas as pd
from sdv.tabular import CTGAN
def generate_synthetic_data(data, model_type, params):
    df = pd.DataFrame(data)
    if model_type == 'CTGAN':
        model = CTGAN(**params)
        model.fit(df)
        synthetic = model.sample(len(df))
    return synthetic
