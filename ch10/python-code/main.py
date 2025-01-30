import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas as pd
except ImportError:
    install('pandas')
    import pandas as pd

import data_processing as dp
import tokenization as tk
import visualization as vz
import training as tr
import evaluation as ev
from transformers import AutoModelForSequenceClassification

# Load and prepare data
train_df = dp.load_data('train.csv')
train_df = dp.prepare_input(train_df)
train_ds = dp.create_dataset(train_df)

# Tokenize data
tok_ds = tk.tokenize_dataset(train_ds)

# Split data
dds = tok_ds.train_test_split(0.25, seed=42)

# Load and prepare evaluation data
eval_df = dp.load_data('test.csv')
eval_df = dp.prepare_input(eval_df)
eval_ds = dp.create_dataset(eval_df).map(tk.tok_func, batched=True)

# Rename 'score' to 'labels' in the datasets
dds = dds.map(lambda x: {'labels': x['score']})
eval_ds = eval_ds.map(lambda x: {'labels': 0}) # Add dummy labels for evaluation

# Initialize model and training arguments
model = AutoModelForSequenceClassification.from_pretrained(tk.model_nm, num_labels=1)
args = tr.create_training_args('outputs', lr=8e-5, bs=128, epochs=4)

# Train model
trainer = tr.train_model(model, args, dds['train'], dds['test'], tk.tokz, ev.corr_d)

# Evaluate model
preds = ev.evaluate_model(trainer, eval_ds)
print(preds)

# Predict and save submission file
preds = trainer.predict(eval_ds).predictions.astype(float)
preds = np.clip(preds, 0, 1)
import datasets

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds
})

submission.to_csv('submission.csv', index=False)
