#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization




# In[35]:


path = "C:/Users/anilk/Jupyter_Projects/imputed_data.csv"
# path = "C:/Users/anilk/Jupyter_Projects/df_one_hot_encoding_with_lasso_feature_selection.txt"
imputed_df = pd.read_fwf(path)

print(imputed_df.head())


# In[28]:


imputed_df['ciaf'] = imputed_df['ciaf'].astype('category').cat.codes

X = imputed_df.drop('ciaf', axis=1)
y = imputed_df['ciaf']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[31]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(60, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(30, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(15, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))
model.add(BatchNormalization())


# In[32]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)


# In[33]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

y_pred_prob = model.predict(X_test)
y_pred = (model.predict(X_test) > 0.5).astype("int32")


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f'ROC-AUC Score: {roc_auc}')



fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[36]:


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')



plt.tight_layout()
plt.show()


# In[ ]:




