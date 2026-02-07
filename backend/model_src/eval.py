import torch

from model_src.prepare_train.metrics import reset_metrics, update_metrics, show_results

def evaluate(predictor, dataloader, train_params, collect_error_analysis=False):
    device = train_params.device
    loss_fn = train_params.loss_fn
    metrics = train_params.metrics

    loss = 0
    reset_metrics(metrics)
    predictor.get_model().to(device)
    predictor.get_model().eval()
    with torch.no_grad():
        for X, y, indices in dataloader:
            if isinstance(X, dict):
                X = {k: v.to(device) for k, v in X.items()}
            else:
                X = X.to(device)
            y = y.to(device)
            logits = predictor.logits(X)
            loss += loss_fn(logits, y).item()
            
            if collect_error_analysis:
                train_params.error_analysis.update(indices, logits, predictor, y)
            
            preds = predictor.preds(logits)
            update_metrics(metrics, preds, y)
    metric_results = show_results(metrics)
    metric_results.append(('loss', loss))
    if collect_error_analysis:
        dict_error_analysis = train_params.error_analysis.get_results()
        return metric_results, dict_error_analysis
    return metric_results, None

def predict(predictor, test_dl, device, metrics, error_analysis):
    reset_metrics(metrics)
    predictor.get_model().to(device)
    predictor.get_model().eval()
    with torch.no_grad():
        for X, y, indices in test_dl:
            if isinstance(X, dict):
                X = {k: v.to(device) for k, v in X.items()}
            else:
                X = X.to(device)
            y = y.to(device)
            logits = predictor.logits(X)
            preds = predictor.preds(logits)
            update_metrics(metrics, preds, y)
            error_analysis.test_update(indices, preds, y)
            
    metric_results = show_results(metrics)
    dict_error_analysis = error_analysis.get_results()
    return metric_results, dict_error_analysis