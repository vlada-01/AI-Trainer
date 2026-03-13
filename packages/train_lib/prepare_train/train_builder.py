from packages.train_lib.prepare_train.meta import create_meta
from packages.train_lib.prepare_train.engines.engine_manager import create_train_manager

def prepare_engine(train_cfg, model, train, val, test, meta):
    train_meta = create_meta(train_cfg, model.parameters(), meta)
    engine = create_train_manager(model, train, val, test, meta)
    return engine, train_meta