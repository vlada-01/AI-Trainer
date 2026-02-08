import re
import torch
from torch.utils.data import DataLoader

from model_src.data.metas.meta import MetaData, MetaTypes

from common.logger import get_logger

log = get_logger(__name__)

class TextualMetaData(MetaData):
    def __init__(self):
        super().__init__(MetaTypes.textual)
        self.task = None
        self.vocab = None
        self.max_len = None
        self.input_size = None
        self.output_size = None

    def update(self, upd_dict):
        for k, v in upd_dict.items():
            if hasattr(self, k):
                if callable(getattr(self, k)):
                    log.info(f'Calling TextualMetaData attr "{k}"')
                    fn = getattr(self, k)
                    if isinstance(v, dict):
                        fn(**v)
                    elif isinstance(v, (list, tuple)):
                        fn(*v)
                    else:
                        fn(v)
                else:
                    raise ValueError(f'TextualMetaData does not have callable {k}')
            else:
                log.warning(f'TextualMetaData does not have attr {k}')

    def resolve(self, name):
        fn = getattr(self, f'resolve_{name.value}', None)
        if fn is None:
            raise NameError(f'Name resolve_{name.value} is not supported in {type(self)}')
        return name, fn()
    
    def get_necessary_sizes(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'vocab_size': len(self.vocab),
            'max_len_size': self.max_len
        }
    
    def get_task(self):
        return self.task
    
    def get_unique_targets(self):
        return int(self.output_size[0])
    
    def to_dict(self):
        return {
            'modality': self.modality,
            'task': self.task,
            'vocab': self.vocab,
            'max_len': self.max_len,
            'input_size': list(self.input_size),
            'output_size': list(self.output_size),
            'task': self.task,
        }
    
    # ---------------------- Resolvers ----------------------

    def resolve_text_to_tensor(self):
        params = {
            'vocab': self.vocab,
            'max_len': self.max_len
        }
        return params
    
    def resolve_to_tensor(self):
        return {}
    
    # ---------------------- Internal -----------------------
    # TODO: 
    # Should improve the vocab logic, by adding max_size
    # add more frequent tokens first
    def prepare_textual_params(self, raw_train, mapper):
        log.info('Preparing vocabulary')
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3,
        }
        vocab_id = 4
        greater_then_attn_mask = 0
        attn_sizes = []
        max_attn_mask = 0
        for i in range(len(raw_train)):
            raw_dict = raw_train[i]
            tokens, _ = mapper(raw_dict)
            if len(tokens) + 2 > self.max_len:
                greater_then_attn_mask += 1
            if len(tokens) + 2 > max_attn_mask:
                max_attn_mask = len(tokens) + 2
            attn_sizes.append(len(tokens) + 2)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = vocab_id
                    vocab_id += 1
        self.vocab = vocab
        log.info(f'Size of vocabulary {len(self.vocab)}')
        
        p90 = int(0.9 * (len(attn_sizes) -1)) 
        new_max_len = sorted(attn_sizes)[p90]
        if new_max_len > self.max_len:
            log.info(f'Overriding current max_len={self.max_len} with new max_len={new_max_len}')
            self.max_len = new_max_len

        log.info(f'Max attention mask: {self.max_len}. Greater then attn_mask: {greater_then_attn_mask}, max size: {max_attn_mask}')
    
    # TODO: need to be careful what is the output.size() - 1 num, list, list[list]
    def set_sizes(self, train_ds):
        log.info('Updating TextualMetaData sizes')
        
        if self.task == 'regression':
            X, y = train_ds[0]
            self.input_size = {
                'input_ids': X['input_ids'].size(),
                'attn_mask': X['attention_mask'].size()
            }
            self.output_size = y.size()
            return

        loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)

        seen = set()
        for batch in loader:
            _, y = batch
            if hasattr(y, "tolist"):
                seen.update(y.tolist())
            else:
                seen.update(list(y))

        num_classes = len(seen)

        log.info(f'Number of classes: {num_classes}')
        
        X, _ = train_ds[0]
        self.input_size = {
            'input_ids': X['input_ids'].size(),
            'attn_mask': X['attention_mask'].size()
        }
        self.output_size = torch.Size([num_classes])

    def set_task(self, task):
        self.task = task

    def set_max_len(self, max_len):
        self.max_len = max_len