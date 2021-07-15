import pathlib
import os
# path where is trained models, loaders and train_dataset
path = str(pathlib.Path().absolute())
root_dir = os.path.dirname(os.path.abspath(__file__))

# host
host = '0.0.0.0'
port = 5000