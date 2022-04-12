from flax.serialization import msgpack_serialize
import jax
import jax.numpy as np
import jax.random as rand
import math
import numpy as onp
from opencc import OpenCC
import optax
from transformers import BertTokenizer, FlaxBertForSequenceClassification

learning_rate = 0.01
batch_size = 256
n_epochs = 100
key = rand.PRNGKey(42)

model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = FlaxBertForSequenceClassification.from_pretrained(model_name)
params = model.params

convert = OpenCC('hk2s').convert

def cross_entropy_loss(logits, labels):
    exp_logits = np.exp(logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    exp_loss = np.take_along_axis(softmax_probs, labels[..., None], axis=-1)
    loss = -np.log(exp_loss)
    return np.sum(loss)

def load_dataset(text_file, label_file):
    sentences = []
    labels = []

    with open(text_file) as f1, open(label_file) as f2:
        for line, label in zip(f1, f2):
            label = int(label)
            line = line.rstrip('\n')
            line = convert(line)
            sentences.append(line)
            labels.append(label)

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='np')
    labels = onp.array(labels, dtype=np.bool_)

    return inputs.input_ids, inputs.attention_mask, labels

def save_params(params, filename):
    serialized_params = msgpack_serialize(params)
    with open(filename, 'wb') as f:
        f.write(serialized_params)

train_inputs, train_masks, train_labels = load_dataset('train.text.txt', 'train.label.txt')
train_size = len(train_inputs)

@jax.jit
@jax.value_and_grad
def forward(params, key, inputs, labels, mask):
    outputs = model(input_ids=inputs, attention_mask=mask, params=params, train=True, dropout_rng=key)
    logits = outputs.logits
    loss = cross_entropy_loss(logits, labels)
    return loss

@jax.jit
def predict(params, inputs, mask):
    outputs = model(input_ids=inputs, attention_mask=mask, params=params)
    logits = outputs.logits
    return logits.argmax(-1).astype(np.bool_)

param_labels = {
    'bert': {
        'embeddings': 'freeze',
        'encoder': 'freeze',
        'pooler': 'train',
    },
    'classifier': 'train',
}
optimizer_scheme = {
    'train': optax.adam(learning_rate=learning_rate),
    'freeze': optax.set_to_zero(),
}
optimizer = optax.multi_transform(optimizer_scheme, param_labels)
opt_state = optimizer.init(params)

permute = lambda key: onp.asarray(jax.jit(lambda key: rand.permutation(key, train_size), backend='cpu')(key))

n_batch = math.ceil(train_size // batch_size)

for epoch in range(1, n_epochs + 1):
    key, subkey = rand.split(key)
    shuffled_indices = permute(subkey)

    total_loss = 0.

    for i in range(n_batch):
        batch_idx = shuffled_indices[i*batch_size:(i+1)*batch_size]
        batch_inputs = train_inputs[batch_idx]
        batch_labels = train_labels[batch_idx]
        batch_masks = train_masks[batch_idx]

        key, subkey = rand.split(key)
        loss, grads = forward(params, subkey, batch_inputs, batch_labels, batch_masks)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        total_loss += loss.item()

    print(f'Epoch {epoch}, total loss {total_loss:.2f}')

save_params(params, 'params.dat')
