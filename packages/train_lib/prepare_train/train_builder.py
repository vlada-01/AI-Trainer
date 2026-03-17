from .meta import create_meta
from .engines import create_train_manager

def build_engine(train_cfg, model, train, val, test, meta):
    train_meta = create_meta(train_cfg, model.parameters(), meta)
    engine = create_train_manager(model, train, val, test, train_meta)
    return engine, train_meta