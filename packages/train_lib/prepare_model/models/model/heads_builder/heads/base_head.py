from abc import ABC, abstractmethod
# FIXME: consider updating this as abstract class because there are functions required from model
class Head:
    def __init__(self, task):
        self.task = task
        self.pps_chain = None

    def get_task(self):
        return self.task
    
    def attach_pps(self, pps_chain):
        self.pps_chain = pps_chain

    def get_pps_chain(self):
        return self.pps_chain

    def collect_samples(self, logits, targets):
        self.pps_chain.collect_samples(logits, targets)

    def fit_pps(self):
        self.pps_chain.fit_pps()

    @abstractmethod
    def process(self, logits, apply_pp, return_details):
        pass
    
    @abstractmethod
    def get_final_out(self, h_out):
        pass

    @abstractmethod
    def get_metrics_out(self, h_out):
        pass

    @abstractmethod
    def get_error_analysis_out(self, h_out):
        pass