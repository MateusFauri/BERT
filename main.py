import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from transformers import BertTokenizer
from transformers import TFBertModel



def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


df = pd.read_csv('data/train.tsv', sep='\t')

seq_len = 256
num_samples = len(df)

Xids = np.zeros((num_samples, seq_len))
Xmask = np.zeros((num_samples, seq_len))

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

for i, phrase in enumerate(df['Phrase']):
    tokens = tokenizer.encode_plus(phrase,
                                   max_length=seq_len,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   return_tensors='tf')
    Xids[i, :] = tokens['input_ids']
    Xmask[i, :] = tokens['attention_mask']

arr = df['Sentiment'].values
print(arr)

labels = np.zeros((num_samples, arr.max() + 1))
labels[np.arange(num_samples), arr] = 1

dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
dataset = dataset.map(map_func)

batch_size = 16
dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)

split = 0.8
size = int((num_samples / batch_size) * split)

print(size)

train_ds = dataset.take(size)
val_ds = dataset.skip(size)

bert = TFBertModel.from_pretrained('bert-base-cased')

input_ids = tf.keras.layers.Input(shape=(seq_len,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(seq_len,), name='attention_mask', dtype='int32')

embeddings = bert.bert(input_ids, attention_mask=mask, )[1]
x = tf.keras.layers.Dense(512, activation='relu')(embeddings)
y = tf.keras.layers.Dense(arr.max() + 1, activation='softmax', name='outputs')(x)

model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
model.layers[2].trainable = False
print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1
)

model.save('sentiment_model')
