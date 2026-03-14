import pandas as pd
import numpy as np

from packages.train_lib.prepare_train.error_tables.error_table import ErrorTable


class RegressionErrorTable(ErrorTable):
    def __init__(self):
        super().__init__()

    def set_states(self, meta, key):
        pass

    def restart_error_table(self):
        self.df = pd.DataFrame()

    def restart(self):
        self.restart_error_table()

    # TODO: need to add post processing for regression to use pp-ed logits
    def update(self, ids, h_outs, targets):
        mandatory_outs = h_outs['mandatory']
        optional_outs = h_outs['optional']

        ids = ids.cpu().numpy()
        targets=targets.cpu().numpy()
        final = mandatory_outs['final'].cpu().numpy()

        error = (final - targets).mean(axis=1)
        abs_error = np.abs(error)
        squared_error = np.square(error)

        new_df = pd.DataFrame({
            'id': ids,
            'y_true': targets,
            'y_pred': final,
            'error': error,
            'abs_error': abs_error,
            'squared_error': squared_error,
            **{k: v.cpu().numpy() for k, v in optional_outs.items()}
        })
        if len(self.df) == 0:
            self.df = new_df.copy()
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)

    def collect_error_table(self):
        return self.df