from flax.serialization import msgpack_restore
import jax
import jax.numpy as np
# import math
# import numpy as onp
from opencc import OpenCC
from transformers import BertConfig, BertTokenizer, FlaxBertForSequenceClassification

model_name = 'bert-base-chinese'
config = BertConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = FlaxBertForSequenceClassification(config=config)

convert = OpenCC('hk2s').convert

def load_params(filename):
    with open(filename, 'rb') as f:
        serialized_params = f.read()
    return msgpack_restore(serialized_params)

model.params = load_params('params.dat')

@jax.jit
def predict(inputs, mask):
    outputs = model(input_ids=inputs, attention_mask=mask)
    logits = outputs.logits
    return logits.argmax(-1).astype(np.bool_)

def classify(sentences):
    sentences = [convert(sentence[:20]) for sentence in sentences]
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='np')
    inputs, mask = inputs.input_ids, inputs.attention_mask
    labels = predict(inputs, mask)
    return labels.tolist()

def compute_accuracy(labels, predicted_labels):
    total = 0
    correct = 0
    for y, y_hat in zip(labels, predicted_labels):
        total += 1
        if y == y_hat:
            correct += 1
    return correct / total

sentences = []
labels = []

with open('test.text.txt') as f1, open('test.label.txt') as f2:
    for line, label in zip(f1, f2):
        label = int(label) > 1
        line = line.rstrip('\n')
        sentences.append(line)
        labels.append(label)

predicted_labels = classify(sentences)

print(f'Acc: {compute_accuracy(labels, predicted_labels) * 100:.2f}%')
