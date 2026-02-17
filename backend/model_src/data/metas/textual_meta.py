import torch
from transformers import AutoTokenizer
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
        self.input_keys = None

    def preprocess_raw(self, ds):
        log.info('Preprocessing Textual datasets with "distilbert-base-uncased" tokenizer')
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
        
        def tokenize_fn(batch):
            texts = batch['x']
            return {
                'x': [tokenizer.tokenize(t) for t in texts]
            }
        
        ds = ds.map(tokenize_fn, batched=True, num_proc=4)
        return ds

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
                    raise ValueError(f'TextualMetaData does not have callable: {k}')
            else:
                log.warning(f'TextualMetaData does not have attr {k}')

    def resolve(self, name):
        fn = getattr(self, f'resolve_{name.value}', None)
        if fn is None:
            raise NameError(f'Name resolve_{name.value} is not supported in {type(self)}')
        return name, fn()
    
    def get_input_keys(self):
        return self.input_keys

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
            'input_size': self.input_size,
            'output_size': self.output_size,
            'input_keys': self.input_keys,
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
    def prepare_textual_params(self, raw_train):
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
            x = raw_dict['x']
            if len(x) + 2 > self.max_len:
                greater_then_attn_mask += 1
            if len(x) + 2 > max_attn_mask:
                max_attn_mask = len(x) + 2
            attn_sizes.append(len(x) + 2)
            for token in x:
                if token not in vocab:
                    vocab[token] = vocab_id
                    vocab_id += 1
        self.vocab = vocab
        log.info(f'Size of vocabulary {len(self.vocab)}')
        
        p80 = int(0.8 * (len(attn_sizes) -1))
        new_max_len = sorted(attn_sizes)[p80]
        if new_max_len > self.max_len:
            log.info(f'Overriding current max_len={self.max_len} with new max_len={new_max_len}')
            self.max_len = new_max_len
        
        # TODO: wrong information if the max_len is overriden
        log.info(f'Max attention mask: {self.max_len}. Greater then attn_mask: {greater_then_attn_mask}, max size: {max_attn_mask}')
    
    # TODO: need to be careful what is the output.size() - 1 num, list, list[list]
    def set_sizes(self, train_ds):
        log.info('Updating TextualMetaData sizes')
        
        if self.task == 'regression':
            sample_dict, _ = train_ds[0]
            X, y = sample_dict['X'], sample_dict['y']
            self.input_size = {
                k: list(v.size()) for k, v in X.items()
            }
            self.output_size = list(y.size())
            return

        loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)

        seen = set()
        for samples, _ in loader:
            y = samples['y']
            if hasattr(y, "tolist"):
                seen.update(y.tolist())
            else:
                seen.update(list(y))

        num_classes = len(seen)

        log.info(f'Number of classes: {num_classes}')
        
        sample_dict, _ = train_ds[0]
        X, y = sample_dict['X'], sample_dict['y']
        self.input_size = {
            k: list(v.size()) for k, v in X.items()
        }
        self.output_size = list(torch.Size([num_classes]))

    def set_task(self, task):
        self.task = task

    def set_max_len(self, max_len):
        self.max_len = max_len

    def set_input_keys(self, sample, id):
        X_dict = sample['X']
        self.input_keys = [k for k in X_dict.keys()]