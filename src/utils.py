import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_COLUMNS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
TARGET_COLUMN = 'Species'
SPECIES_NAMES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
FEATURE_LABELS = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']

def get_project_root():
    """Returns the project root path."""
    return PROJECT_ROOT

def get_data_path():
    """Returns the path to data/Iris.csv."""
    path = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, 'Iris.csv')

def get_output_path(filename):
    """Returns the path inside outputs/graphs/ for the given filename."""
    path = os.path.join(PROJECT_ROOT, 'outputs', 'graphs')
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)

def get_model_path():
    """Returns the path to saved_model/best_model.pkl."""
    path = os.path.join(PROJECT_ROOT, 'saved_model')
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, 'best_model.pkl')

def load_dataset():
    """Loads Iris.csv, drops 'Id' column, returns DataFrame."""
    df = pd.read_csv(get_data_path())
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    return df

def setup_plot_style():
    """Sets matplotlib/seaborn professional style."""
    sns.set_style("whitegrid")
    sns.set_palette("muted") # Professional color palette
    plt.rcParams['figure.dpi'] = 150
