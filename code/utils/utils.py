import evaluate

def compute_metrics(pred):
    roc_auc = evaluate.load("roc_auc")
    logits, labels = pred
    probs = logits[:, 1]  
    auc = roc_auc.compute(prediction_scores=probs, references=labels)["roc_auc"]  
    return {"auroc": auc}