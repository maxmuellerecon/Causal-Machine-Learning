#Test file for I_Double_machine_learning

import pytest
pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.I_Double_machine_learning import plot_pattern, fwl_theorem, verify_fwl_theorem, debias_treatment, denoise_outcome, comparison_models, parametric_double_ml_cate, non_parametric_double_ml_cate, orthogonalize_treatment_and_outcome

def test_plot_pattern_type_input():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    plot = plot_pattern(train, "price", "sales")
    assert isinstance(train, pd.DataFrame)
    
def test_fwl_theorem_type_input():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    table1 = fwl_theorem(train)
    assert isinstance(train, pd.DataFrame)
    
def test_verify_fwl_theorem_type_input():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    table2 = verify_fwl_theorem(train)
    assert isinstance(train, pd.DataFrame)
    
def test_debias_treatment_type_output():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    train_pred = debias_treatment(train)
    assert isinstance(train_pred, pd.DataFrame)
    
def test_debias_treatment_shape_output():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    train_pred = debias_treatment(train)
    assert train_pred.shape == (10000, 6)
    
def test_denoise_outcome_type_output():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    train_pred = pd.read_csv(BLD / "python" / "Lesson_I" / "data" / "train_pred.csv")
    train_pred_y = denoise_outcome(train, train_pred)
    assert isinstance(train_pred_y, pd.DataFrame)
    
def test_denose_outcome_shape_output():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    train_pred = pd.read_csv(BLD / "python" / "Lesson_I" / "data" / "train_pred.csv")
    train_pred_y = denoise_outcome(train, train_pred)
    assert train_pred_y.shape == (10000, 7)
    
def test_comparison_models_type_input_pred():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    train_pred_y = pd.read_csv(BLD / "python" / "Lesson_I" / "data" / "train_pred_y.csv")
    final_model, basic_model = comparison_models(train_pred_y, train)
    assert isinstance(train_pred_y, pd.DataFrame)
    
def test_comparison_models_type_input_train():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    train_pred_y = pd.read_csv(BLD / "python" / "Lesson_I" / "data" / "train_pred_y.csv")
    final_model, basic_model = comparison_models(train_pred_y, train)
    assert isinstance(train, pd.DataFrame)

def test_parametric_double_ml_cate_type_output_data():
    train_pred_y = pd.read_csv(BLD / "python" / "Lesson_I" / "data" / "train_pred_y.csv")
    test = pd.read_csv(SRC / "data" / "ice_cream_sales_rnd.csv")
    final_model_cate, cate_test = parametric_double_ml_cate(train_pred_y, test)
    assert isinstance(cate_test, pd.DataFrame)
    
def test_orthogonalize_treatment_and_outcome_type_output():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    train_pred_nonparam = orthogonalize_treatment_and_outcome(train)
    assert isinstance(train_pred_nonparam, pd.DataFrame)
    
def test_orthogonalize_treatment_and_outcome_type_shape():
    train = pd.read_csv(SRC / "data" / "ice_cream_sales.csv")
    train_pred_nonparam = orthogonalize_treatment_and_outcome(train)
    assert train_pred_nonparam.shape == (10000, 7)
    
def test_non_parametric_double_ml_cate_type_output():
    train_pred_nonparam = pd.read_csv(BLD / "python" / "Lesson_I" / "data" / "train_pred_nonparam.csv")
    test = pd.read_csv(SRC / "data" / "ice_cream_sales_rnd.csv")
    cate_test_nonparam = non_parametric_double_ml_cate(train_pred_nonparam, test)
    assert isinstance(cate_test_nonparam, pd.DataFrame)
    
def test_non_parametric_double_ml_cate_shape_output():
    train_pred_nonparam = pd.read_csv(BLD / "python" / "Lesson_I" / "data" / "train_pred_nonparam.csv")
    test = pd.read_csv(SRC / "data" / "ice_cream_sales_rnd.csv")
    cate_test_nonparam = non_parametric_double_ml_cate(train_pred_nonparam, test)
    assert cate_test_nonparam.shape == (5000, 6)