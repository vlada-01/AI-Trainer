import pandas as pd
import numpy as np

from packages.train_lib.prepare_train.error_tables.error_analysis_manager import ErrorTable

class ClassificationErrorTable(ErrorTable):
    def __init__(self):
        super().__init__()
        # extras
        self.num_classes = None
        self.confusion_matrix = None

    def set_states(self, meta, key):
        data_meta = meta.get_data_meta()
        uniques = data_meta.get_output_unique_values(key)
        self.num_classes = uniques
        self.confusion_matrix = np.zeros((uniques, uniques + 1))

    def restart_error_table(self):
        self.df = pd.DataFrame() 

    def restart(self):
        self.restart_error_table()
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes + 1))
    
    def update(self, ids, h_outs, targets):
        mandatory_outs = h_outs['mandatory']
        optional_outs = h_outs['optional']

        ids = ids.cpu().numpy()
        targets = targets.cpu().numpy()
        probs = mandatory_outs['probs'].cpu().numpy()
        final = mandatory_outs['final'].cpu().numpy()
        
        is_raw_correct = (final == targets)
        confidence = np.max(probs, axis=1)
        y_true_prob = probs[np.arange(len(targets)), targets]
        
        new_df = pd.DataFrame({
            'id': ids,
            'y_true': targets,
            'y_pred': final,
            'is_correct': is_raw_correct,
            'confidence': confidence,
            'y_true_prob': y_true_prob,
            **{k: v.cpu().numpy() for k, v in optional_outs.items()}
        })

        np.add.at(self.confusion_matrix, (targets, final), 1)
        if len(self.df) == 0:
            self.df = new_df.copy()
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)

    def collect_error_table(self):
        return self.df

    def collect_extras(self):
        return {
            'confusion_matrix': self.confusion_matrix.tolist()
        }