from packages.train_lib.prepare_train.engines.train_engine.engine import prepare_train_engine, TrainEngine
from packages.train_lib.prepare_train.engines.eval_engine.engine import prepare_eval_engine, EvaluationEngine

from packages.logger.logger import get_logger

log = get_logger(__name__)

def create_train_manager(model, train, val, test, meta):
    log.info('Initializing EngineManager')
    engine = EngineManager(model, train, val, test, meta)
    log.info('EngineManager successfully prepared')
    return engine

class EngineManager:
    def __init__(self, model, train, val, test, meta):
        self.model = model
        self.train = train
        self.val = val
        self.test = test
        self.train_engine: TrainEngine = prepare_train_engine(meta)
        self.eval_engine: EvaluationEngine = prepare_eval_engine(meta)

    def train_model(self):
        return self.train_engine.train_model(self.model, self.train, self.val)
    
    def train_pp(self):
        return self.train_engine.train_pp(self.model, self.val)

    def evaluate_val(self):
        return self.eval_engine.evaluate(self.model, self.val)

    def evaluate_test(self):
        return self.eval_engine.evaluate(self.model, self.test, return_details=True)

    def get_model(self):
        return self.model