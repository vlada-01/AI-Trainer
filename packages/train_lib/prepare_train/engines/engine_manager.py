from .train_engine.engine import prepare_train_engine, TrainEngine
from .eval_engine.engine import prepare_eval_engine, EvaluationEngine

from packages.logger import get_logger

log = get_logger(__name__)

def create_train_manager(model, train, val, test, train_meta):
    log.info('Initializing EngineManager')
    engine = EngineManager(model, train, val, test, train_meta)
    log.info('EngineManager successfully prepared')
    return engine

class EngineManager:
    def __init__(self, model, train, val, test, train_meta):
        self.model = model
        self.train = train
        self.val = val
        self.test = test
        self.train_engine: TrainEngine = prepare_train_engine(train_meta)
        self.eval_engine: EvaluationEngine = prepare_eval_engine(train_meta)

    def train_model(self, writer):
        return self.train_engine.train_model(self.model, self.train, self.val, writer)
    
    def train_pp(self):
        return self.train_engine.train_pp(self.model, self.val)

    def evaluate_val(self, artifact_writer):
        return self.eval_engine.evaluate(self.model, self.val, artifact_writer)

    def evaluate_test(self, artifact_writer):
        return self.eval_engine.evaluate(self.model, self.test, artifact_writer, return_details=True)

    def get_model(self):
        return self.model