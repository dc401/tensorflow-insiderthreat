import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from pandas.api.types import CategoricalDtype

#Use Pandas to create a dataframe
#In windows to get file from path other than same run directory see:
#https://stackoverflow.com/questions/16952632/read-a-csv-into-pandas-from-f-drive-on-windows-7

URL = 'https://raw.githubusercontent.com/dc401/tensorflow-insiderthreat/master/scenario2-training-dataset-transformed-tf.csv'
dataframe = pd.read_csv(URL)
#print(dataframe.head())

#show dataframe details for column types
#print(dataframe.info())

#print(pd.unique(dataframe['user']))
#https://pbpython.com/categorical-encoding.html
dataframe["user"] = dataframe["user"].astype('category')
dataframe["source"] = dataframe["source"].astype('category')
dataframe["action"] = dataframe["action"].astype('category')
dataframe["user_cat"] = dataframe["user"].cat.codes
dataframe["source_cat"] = dataframe["source"].cat.codes
dataframe["action_cat"] = dataframe["action"].cat.codes

#print(dataframe.info())
#print(dataframe.head())

#save dataframe with new columns for future datmapping
dataframe.to_csv('dataframe-export-allcolumns.csv')

#remove old columns
del dataframe["user"]
del dataframe["source"]
del dataframe["action"]
#restore original names of columns
dataframe.rename(columns={"user_cat": "user", "source_cat": "source", "action_cat": "action"}, inplace=True)
print(dataframe.head())
print(dataframe.info())

#save dataframe cleaned up
dataframe.to_csv('dataframe-export-int-cleaned.csv')


#Split the dataframe into train, validation, and test
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#Create an input pipeline using tf.data
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('insiderthreat')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


#choose columns needed for calculations (features)
feature_columns = []
for header in ["vector", "date", "user", "source", "action"]:
    feature_columns.append(feature_column.numeric_column(header))

#create feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#set batch size pipeline
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#create compile and train model
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)