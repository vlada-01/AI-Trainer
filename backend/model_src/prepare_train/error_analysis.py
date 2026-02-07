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
    if meta.get_task() == 'classification':
        num_classes = meta.get_unique_targets()
        return ClassificationErrorAnalysis(num_classes)
    return RegressionErrorAnalysis()

class RegressionErrorAnalysis:
    def __init__(self):
        self.df = pd.DataFrame(columns=[
            AvailableRegressionColumns.id.value,
            AvailableRegressionColumns.y_true.value,
            AvailableRegressionColumns.y_pred.value,
            AvailableRegressionColumns.error.value,
            AvailableRegressionColumns.abs_error.value,
            AvailableRegressionColumns.relative_error.value
        ])

    def update(self, ids, preds, targets):

        error = preds - targets
        abs_error = torch.abs(error)
        squared_error = torch.square(error)
        relative_error = error / (torch.abs(targets) + 1e-8)

        new_df = pd.DataFrame({
            AvailableRegressionColumns.id.value: ids.detach().cpu().numpy(),
            AvailableRegressionColumns.y_true.value: targets.detach().cpu().numpy(),
            AvailableRegressionColumns.y_pred.value: preds.detach().cpu().numpy(),
            AvailableRegressionColumns.error.value: error.detach().cpu().numpy(),
            AvailableRegressionColumns.abs_error.value: abs_error.detach().cpu().numpy(),
            AvailableRegressionColumns.squared_error.value: squared_error.detach().cpu().numpy(),
            AvailableRegressionColumns.relative_error.value: relative_error.detach().cpu().numpy()
        })
        if len(self.df) == 0:
            self.df = new_df.copy()
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)

    # TODO: needs to be implemented
    def test_update(self, ids, pp_preds, targets):
        pass

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

        new_df, final = predictor.preds_with_error_analysis(logits, new_df)
        np.add.at(self.confusion_matrix, (targets.detach().cpu().numpy(), final.detach().cpu().numpy()), 1)
        
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