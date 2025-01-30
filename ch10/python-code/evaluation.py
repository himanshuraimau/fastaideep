import numpy as np

def corr(x, y):
    return np.corrcoef(x, y)[0][1]

def corr_d(eval_pred):
    return {'pearson': corr(*eval_pred)}

def evaluate_model(trainer, eval_ds):
    preds = trainer.predict(eval_ds).predictions.astype(float)
    return preds
