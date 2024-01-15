#Test file for D_Decison_trees

from sklearn.tree import DecisionTreeRegressor  
import numpy as np
import pytest
pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
import pickle
import graphviz

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.D_Decision_trees import create_data, fit_tree, predict_new_value, visualize_decision_tree

def test_create_data_type():
    data = create_data()
    assert isinstance(data, pd.DataFrame)

def test_create_data_shape():
    data = create_data()
    assert data.shape == (14, 3)
    
def test_fit_tree_is_regressor():
    data = create_data()
    regressor = fit_tree(data)
    assert isinstance(regressor, DecisionTreeRegressor)

def test_predict_new_value():
    data = create_data()
    regressor = fit_tree(data)
    new_value = 3750
    y_pred_value = predict_new_value(regressor, new_value)
    y_pred = regressor.predict(np.array(new_value).reshape(-1, 1))
    assert y_pred == np.array([8000.])
    
def test_visualize_decision_tree():
    regressor = pickle.load(open(BLD / "python" / "Lesson_D" / "model" / "regressor.pkl", 'rb'))
    # Create sample feature names
    feature_names = ['Production Cost']
    # Call the visualize_decision_tree function
    graph = visualize_decision_tree(regressor, feature_names)
    # Check if the returned graph is an instance of graphviz.Source
    assert isinstance(graph, graphviz.Source)