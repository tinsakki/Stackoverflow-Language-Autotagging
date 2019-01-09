from keras.models import model_from_json
import pickle
from pprint import pprint

import numpy as np
with open('model_stack.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("model_stack.h5")
with open('tokenizer123.pickle', 'rb') as handle:
    t = pickle.load(handle)
with open('d.pickle', 'rb') as handle:
    c = pickle.load(handle)

a = [""" always thought Java was pass-by-reference.

However, I've seen a couple of blog posts (for example, this blog) that claim that it isn't.

I don't think I understand the distinction they're making.

What is the explanation?"""]

ans = ''
text = np.array(a)
result = t.texts_to_matrix(text)
result2 = model.predict(result)
prediction = np.argmax(result2)
for k,j in c.items():
    if j==prediction:
        ans = k
print(ans)

