import pandas as pd
import numpy as np
import torch
from enum import Enum

from common.logger import get_logger

log = get_logger(__name__)

class AvailableRegressionColumns(str, Enum):
    id = 'id'
    y_true = 'y_true'
    y_pred = 'y_pred'
    error = 'error'
    abs_error = 'abs_error'
    squared_error = 'squared_error'
    relative_error = 'relative_error' #y_true - y_pred / (abs(y_true) + eps)

class AvailableClassificationColumns(str, Enum):
    id = 'id'
    y_true = 'y_true'
    y_pred = 'y_pred'
    is_correct = 'is_correct'
    confidence = 'confidence'
    y_true_prob = 'y_true_prob'

def prepare_error_analysis(meta):
    error_analysis = {}
    for out_key, task in meta.get_tasks():
        if task == 'classification': 
            error_analysis[out_key] = ClassificationErrorAnalysis(meta.get_output_unique_values(out_key))
        elif task == 'regression':
            error_analysis[out_key] = RegressionErrorAnalysis()
        else:
            raise ValueError(f'Invalid task: {task}')
    
    return error_analysis

def update_error_analysis(error_analysis, ids, logits, predictor, targets):
    for k, v in error_analysis.items():
        curr_logits = logits[k]
        curr_targets = targets[k]
        v.update(ids, curr_logits, predictor, curr_targets)
    return error_analysis

def test_update_error_analysis(error_analysis, ids, preds, targets):
    for k, v in error_analysis.items():
        curr_preds = preds[k]
        curr_targets = targets[k]
        v.test_update(ids, curr_preds, curr_targets)
    return error_analysis

def get_results(error_analysis):
    results = {}
    for k, v in error_analysis.items():
        results[k] = v.get_results()
    return results

class RegressionErrorAnalysis:
    def __init__(self):
        self.df = pd.DataFrame()

    # TODO: need to add post processing for regression to use pp-ed logits
    def update(self, ids, logits, predictor, targets):

        error = (logits - targets).mean(1)
        abs_error = torch.abs(error)
        squared_error = torch.square(error)

        new_df = pd.DataFrame({
            AvailableRegressionColumns.id.value: ids.detach().cpu().numpy(),
            AvailableRegressionColumns.y_true.value: targets.detach().cpu().numpy(),
            AvailableRegressionColumns.y_pred.value: logits.detach().cpu().numpy(),
            AvailableRegressionColumns.error.value: error.detach().cpu().numpy(),
            AvailableRegressionColumns.abs_error.value: abs_error.detach().cpu().numpy(),
            AvailableRegressionColumns.squared_error.value: squared_error.detach().cpu().numpy(),
        })
        if len(self.df) == 0:
            self.df = new_df.copy()
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)

    # TODO: needs to be implemented
    def test_update(self, ids, preds, targets):
        ids = ids.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        error = (preds - targets).mean(1)
        abs_error = torch.abs(error)
        squared_error = torch.square(error)
        new_df = pd.DataFrame({
            AvailableClassificationColumns.id.value: ids,
            AvailableClassificationColumns.y_true.value: targets,
            AvailableClassificationColumns.y_pred.value: preds,
            AvailableRegressionColumns.error.value: error,
            AvailableRegressionColumns.abs_error.value: abs_error,
            AvailableRegressionColumns.squared_error.value: squared_error
        })

        if len(self.df) == 0:
            self.df = new_df.copy()
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)

    def get_results(self):
        return {'df': self.df.to_dict(orient="records")}

class ClassificationErrorAnalysis:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.df = pd.DataFrame()
        
        self.confusion_matrix = np.zeros((num_classes, num_classes + 1))
    
    def update(self, ids, logits, predictor, targets):
        raw_probs = torch.softmax(logits, dim=1)
        raw_preds = torch.argmax(raw_probs, dim=1)
        
        is_raw_correct = (raw_preds == targets)
        confidence = torch.max(raw_probs, dim=1).values
        y_true_prob = raw_probs[torch.arange(len(targets)), targets]
        
        new_df = pd.DataFrame({
            AvailableClassificationColumns.id.value: ids.detach().cpu().numpy(),
            AvailableClassificationColumns.y_true.value: targets.detach().cpu().numpy(),
            AvailableClassificationColumns.y_pred.value: raw_preds.detach().cpu().numpy(),
            AvailableClassificationColumns.is_correct.value: is_raw_correct.detach().cpu().numpy(),
            AvailableClassificationColumns.confidence.value: confidence.detach().cpu().numpy(),
            AvailableClassificationColumns.y_true_prob.value: y_true_prob.detach().cpu().numpy()
        })

        # new_df, final = predictor.preds_with_error_analysis(logits, new_df) Not implemented with pp
        np.add.at(self.confusion_matrix, (targets.detach().cpu().numpy(), raw_preds.detach().cpu().numpy()), 1)
        
        if len(self.df) == 0:
            self.df = new_df.copy()
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
    
    def test_update(self, ids, preds, targets):
        ids = ids.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        is_correct = (preds == targets)
        new_df = pd.DataFrame({
            AvailableClassificationColumns.id.value: ids,
            AvailableClassificationColumns.is_correct.value: is_correct,
            AvailableClassificationColumns.y_true.value: targets,
            AvailableClassificationColumns.y_pred.value: preds,
        })
        
        # pp_preds[pp_preds == -1] = self.num_classes
        np.add.at(self.confusion_matrix, (targets, preds), 1)

        if len(self.df) == 0:
            self.df = new_df.copy()
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)

    def get_results(self):
        return {
            'df': self.df.to_dict(orient="records"),
            'confusion_matrix': self.confusion_matrix.tolist()
        }