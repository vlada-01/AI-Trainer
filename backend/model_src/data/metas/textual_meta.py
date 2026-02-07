import re
from torch.utils.data import DataLoader
from model_src.data.metas.meta import MetaData, MetaTypes

from common.logger import get_logger

log = get_logger(__name__)

class TextualMetaData(MetaData):
    def __init__(self):
        super().__init__(MetaTypes.textual)
        self.task = None
        self.vocab = None
        self.tokenizer = None
        self.attention_mask = None
        self.input_size = None
        self.output_size = None

    def update(self, upd_dict):
        for k, v in upd_dict.items():
            if hasattr(self, k):
                if callable(getattr(self, k)):
                    log.info(f'Calling TextualMetaData attr {k}')
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
    
    def get_sample_size(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size
        }
    
    def get_task(self):
        return self.task
    
    def get_unique_targets(self):
        return self.output_size
    
    # TODO: update me
    def to_dict(self):
        return {
            'modality': self.modality,
            'extras': self.extras,
            'input_size': list(self.input_size),
            'output_size': self.output_size, #TODO: convert to list
            'task': self.task,
        }
    
    # ---------------------- Resolvers ----------------------

    def resolve_text_to_tensor(self):
        params = {
            'tokenizer': self.tokenizer,
            'vocab': self.vocab,
            'attention_mask': self.attention_mask
        }
        return params
    
    def resolve_to_tensor(self):
        return {}
    
    # ---------------------- Internal -----------------------

    def prepare_textual_params(self, raw_train, mapper):
        log.info(f'Preparing tokenizer')
        self.tokenizer = re.compile(r'\w+|[^\w\s]', flags=re.UNICODE)
        
        log.info('Preparing vocabulary')
        vocab = {}
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        vocab['<sos>'] = 2
        vocab['<eos>'] = 3
        vocab_id = 4
        bigger_then_attn = 0
        for i in range(len(raw_train)):
            raw_dict = raw_train[i]
            X, _ = mapper(raw_dict)
            tokens = self.tokenizer.findall(X)
            if len(tokens) + 2 > self.attention_mask:
                bigger_then_attn += 1
            for token in tokens:
                if token not in vocab:
                    vocab[token] = vocab_id
                    vocab_id += 1
        self.vocab = vocab
        log.info(f'Size of vocabulary {len(self.vocab)}')

        log.info(f'Number of samples greater then attn_mask({self.attention_mask}), {bigger_then_attn}')
    
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
        self.output_size = num_classes

    def set_task(self, task):
        self.task = task

    def set_attention_mask(self, attn_mask):
        self.attention_mask = attn_mask