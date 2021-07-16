import numpy as np
import pandas as pd
import algorithms.challenge_senai.predictor_train as pt
import repository.repository_service as rs
row = [-0.62, -1.63, -3.05, -1.37, -0.62, 0.82, 0.82, -1.63, -1.37, -2.52, -1.07, 0.00, -0.90, -1.01, -0.62, 0.02, -1.01, -0.74, 1.86, 5.28]
column_names = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15',
                        'c16', 'c17', 'c18', 'c19', 'c20']

df2 = pd.DataFrame(np.array([row]), columns=column_names)
df2 = df2.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14','c16', 'c18', 'c19']]
row = [df2.iloc[0]]
#row2 = [[-0.62,-1.63,-3.05,-1.37,-0.62,0.82,0.82,-1.63,-1.37,-2.52,-1.07,0.0,-0.9,-1.01,-0.62,0.02,-1.01,-0.74,1.86,5.28]]
stacking = rs.load_model("models_to_evaluate", "level_2", "stacking_ensemble_of_ensemble")

target = stacking.predict(row)
