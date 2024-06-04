import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Debug: Check class distribution
unique, counts = np.unique(labels, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

max_len = max(len(sublist) for sublist in data)
padded_data = [sublist + [0] * (max_len - len(sublist)) for sublist in data]

data = np.array(padded_data)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
score = accuracy_score(y_pred, y_test)

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print(f'{score*100:.2f}% of samples were classified correctly!')
