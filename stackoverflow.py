import pandas as pd
from sklearn.model_selection import KFold,GridSearchCV
from keras.layers.embeddings import Embedding
import pickle
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from pprint import pprint
import pprint
from keras.optimizers import Adam
a = pd.read_csv('stack-overflow-data.csv',dtype=str)
a = a.sample(frac=1)
X = a["post"].astype(str)
from nltk.corpus import stopwords
b = set(stopwords.words('english'))
X = X.apply(lambda x: ' '.join([word for word in x.split() if word not in (b)]))
y = a["tags"].astype(str)
t = Tokenizer()
t.fit_on_texts(X)
encoded = t.texts_to_matrix(X)
with open('tokenizer123.pickle', 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
d = {}
label = LabelEncoder()
Z = label.fit_transform(y)
for i,j in enumerate(y):
    d[j] = Z[i]
Y_change = to_categorical(Z,num_classes=len(y),dtype=np.int8)
def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion)
    print(ratio)
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = shuffle(encoded, Y_change, 3)
model = Sequential()
model.add(Dense(128,input_shape=(X_train.shape[1],),activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(len(y),activation='softmax',kernel_initializer='normal'))
opt = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=100,epochs=50)
ans = model.evaluate(X_test,Y_test)
model_json = model.to_json()
with open("model_stack.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_stack.h5")
print("Saving model to disk")
a = "Python is best language for AI "
text = np.array([a])
result = t.texts_to_matrix(text)
a = model.predict_classes(result)
result2 = model.predict(result)
prediction = np.argmax(result2)
for i,j in d.items():
    if j==prediction:
        print(i)
with open('d.pickle', 'wb') as handles:
    pickle.dump(d, handles, protocol=pickle.HIGHEST_PROTOCOL)
