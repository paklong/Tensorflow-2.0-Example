import tensorflow as tf
from tensorflow import keras, feature_column
from tensorflow.keras import layers, optimizers
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split


Num = 5000
data = {'x1':[random.randint(1, 10) for i in range(Num)],
	'x2':[random.randint(1, 10) for i in range(Num)],
	'x3':[random.randint(1, 10) for i in range(Num)],
	'x4':[random.choice(['A', 'B', 'C']) for i in range(Num)]}

data_pre = {'x1':[1], 'x2':[5], 'x3':[9], 'x4':['A'], 'y': [0]}

df = pd.DataFrame(data)
df_pre = pd.DataFrame(data_pre)

df['y'] = df['x1'] + df['x2'] + df['x3'] + df['x4'].apply(lambda x: 5 if x == 'A' else (10 if x == 'B' else 15) )

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(df, test_size=0.2)


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('y')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

x1 = feature_column.numeric_column("x1")
x2 = feature_column.numeric_column("x2")
x3 = feature_column.numeric_column("x3")
x4 = feature_column.categorical_column_with_vocabulary_list('x4', ['A', 'B', 'C'])
x4_one_hot = tf.feature_column.indicator_column(x4)

feature_columns = [x1, x2, x3, x4_one_hot]
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


batch_size = 1
train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val)
test_ds = df_to_dataset(test)
df_pre = df_to_dataset(df_pre)


model = tf.keras.Sequential([
	feature_layer ,
	layers.Dense(24, activation='softmax'),
	layers.Dense(12, activation='linear'),
	layers.Dense(1, activation='linear')
	])


model.compile(optimizer= tf.keras.optimizers.Adam(0.01),
              loss= 'mse',
              metrics= ['mse'])

model.fit(train_ds, validation_data=val_ds, epochs=5)


print (model.predict(df_pre))
