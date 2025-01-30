import torch
from transformers import TrainingArguments, Trainer

def create_training_args(output_dir, lr, bs, epochs):
    return TrainingArguments(
        output_dir, learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=0.01, report_to='none'
    )

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
        return (loss, outputs) if return_outputs else loss

def train_model(model, args, train_dataset, eval_dataset, tokenizer, compute_metrics):
    trainer = RegressionTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
