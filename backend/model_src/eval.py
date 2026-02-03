import torch

from model_src.prepare_train.metrics import reset_metrics, update_metrics, show_results

def evaluate(model, dataloader, train_params, collect_error_analysis=False, post_processor=None):
    device = train_params.device
    loss_fn = train_params.loss_fn
    metrics = train_params.metrics

    loss = 0
    reset_metrics(metrics)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for X, y, indices in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss += loss_fn(preds, y).item()
           
            if collect_error_analysis:
                train_params.error_analysis.update(indices, preds, y, post_processor)
            update_metrics(metrics, preds, y, post_processor)
    metric_results = show_results(metrics)
    metric_results.append(('loss', loss))
    if collect_error_analysis:
        dict_error_analysis = train_params.error_analysis.get_results()
        return metric_results, dict_error_analysis
    return metric_results, None

def predict(model, test_dl, device, metrics, error_analysis, post_processor):
    reset_metrics(metrics)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for X, y, indices in test_dl:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            error_analysis.test_update(indices, preds, y, post_processor)
            update_metrics(metrics, preds, y, post_processor)
    metric_results = show_results(metrics)
    dict_error_analysis = error_analysis.get_results()
    return metric_results, dict_error_analysis