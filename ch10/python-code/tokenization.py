from transformers import AutoTokenizer

model_nm = 'microsoft/deberta-v3-small'
tokz = AutoTokenizer.from_pretrained(model_nm)

def tokenize_input(text):
    return tokz.tokenize(text)

def tok_func(x):
    return tokz(x["input"])

def tokenize_dataset(ds):
    return ds.map(tok_func, batched=True)
