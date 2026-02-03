from model_src.data.metas.meta import MetaData, MetaTypes
class ImageMetaData(MetaData):
    def __init__(self, task, output=None, input=None, extras=None, tr_sample=None, val_sample=None, test_sample=None):
        super().__init__(MetaTypes.image, extras, tr_sample, val_sample, test_sample)
        self.task = task
        self.input = input
        self.output = output

    def resolve(self, name):
        fn = getattr(self, f'resolve_{name.value}', None)
        if fn is None:
            raise NameError(f'Name resolve_{name.value} is not supported in {type(self)}')
        return name, fn()
    
    def add_sample_info(self, train, val, test):
        self.train_sample = train
        self.val_sample = val
        self.test_sample = test
    
    def get_sample_sizes(self):
        return {
            'sample': self.get_sample_size()
            # 'train_sample': self.get_sample_size(self.train_sample),
            # 'val_sample': self.get_sample_size(self.val_sample),
            # 'test_sample': self.get_sample_size(self.test_sample)
        }
    
    def get_task(self):
        return self.task
    
    # TODO: implement
    def get_unique_targets(self):
        return self.output
    
    def to_dict(self):
        return {
            'modality': self.modality,
            'extras': self.extras,
            'input': list(self.input),
            'output': self.output, #TODO: convert to list
            'task': self.task,
        }
    
    def get_sample_size(self):
        # if sample is None:
        #     return None 
        
        return {
            'input_size': self.input,
            'output_size': self.output
        }

    # ---------------------- Resolvers ----------------------
    
    def resolve_img_to_tensor(self):
        return {}
    
    def resolve_to_tensor(self):
        return {}