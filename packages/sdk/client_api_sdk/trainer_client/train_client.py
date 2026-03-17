import yaml
import requests

from .run_client import RunClient

# TODO: this client shall handle exceptions from server appropriately and do something with that
class TrainerClient:
    def __init__(self, server_url=None):
        self.server_url = server_url or 'http://localhost:8000'
        self.run_client = RunClient()

    def set_url(self, server_url):
        self.server_url = server_url

    def get_run_info(self, run_id):
        info_url = f'{self.server_url}/{run_id}'
        return self.run_client.get_info(info_url)
    
    def get_job_info(self, run_id, job_id):
        info_url = f'{self.server_url}/{run_id}/{job_id}'
        return self.run_client.get_info(info_url)

    @staticmethod
    def load_cfg(file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    # can run fail in init
    def start_run(self, file_path):
        run_cfg = self.load_cfg(file_path)
        run_url = f'{self.server_url}/runs'
        return self.run_client.start_run(url=run_url, payload=run_cfg)

    def prepare_dataset(self, run_id, file_path):
        dataset_cfg = self.load_cfg(file_path)
        data_url = f'{self.server_url}/{run_id}/prepare-jobs/dataset'
        return self.run_client.request_job(run_id, url=data_url, payload=dataset_cfg)

    def prepare_model(self, run_id, file_path):
        model_cfg = self.load_cfg(file_path)
        model_url = f'{self.server_url}/{run_id}/prepare-jobs/model'
        return self.run_client.request_job(run_id, url=model_url, payload=model_cfg)

    def prepare_engine(self, run_id, file_path):
        engine_cfg = self.load_cfg(file_path)
        engine_url = f'{self.server_url}/{run_id}/prepare-jobs/engine'
        return self.run_client.request_job(run_id, url=engine_url, payload=engine_cfg)
    
    #FIXME: add post process, final eval, load run, load complete,

    def start_train(self, run_id, exp_name, run_name, model_name):
        train_dict = {
            'exp_name': exp_name,
            'run_name': run_name,
            'model_name': model_name
        }
        train_url = f'{self.server_url}/{run_id}/exec-jobs/train'
        return self.run_client.request_job(run_id, url=train_url, payload=train_dict)
    
    def get_mlflow_history(self):
        mlflow_url = f'{self.server_url}/mlflow/history'
        resp = requests.get(mlflow_url)
        resp.raise_for_status()
        return resp.json()
    
    def get_exp_runs(self, exp_name):
        mlflow_url = f'{self.server_url}/mlflow/get-exp-runs/{exp_name}'
        resp = requests.get(mlflow_url)
        resp.raise_for_status()
        return resp.json()
    
    # need to implement get_mlflow_run details
    
    def delete_mlflow_run(self, mlflow_id):
        mlflow_url = f'{self.server_url}/mlflow/{mlflow_id}'
        resp = requests.delete(mlflow_url)
        resp.raise_for_status()
        return resp.json()
    
    def get_dataset_info(self, id, name=None):
        ds_info_url = f'{self.server_url}/data-info'
        payload = {'id': id, 'name': name}
        resp = requests.post(ds_info_url, json=payload)
        resp.raise_for_status()
        return resp.json()


