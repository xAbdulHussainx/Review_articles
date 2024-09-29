import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

train_file_path = 'train.crdownload'  
data = pd.read_csv(train_file_path, delimiter='\t', header=None, names=['Label', 'Review'])

data_clean = data.dropna(subset=['Review']).reset_index(drop=True)

X_train, X_val, y_train, y_val = train_test_split(data_clean['Review'], data_clean['Label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(ngram_range=(1, 2))  

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_val_vec)

test_file_path = 'test.dat'  
with open(test_file_path, 'r') as file:
    test_lines = file.readlines()

test_reviews = pd.DataFrame(test_lines, columns=['Review'])

X_test_vec = vectorizer.transform(test_reviews['Review'])

test_predictions = clf.predict(X_test_vec)

test_output = pd.DataFrame(test_predictions, columns=['Label'])
test_output.to_csv('output5.csv', index=False, header=False)
