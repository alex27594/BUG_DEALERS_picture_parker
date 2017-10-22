import sys
import pickle

countVectorizer = pickle.load(open("vectorizer_description", "rb"))
clf = pickle.load(open("clf_description", "rb"))

query = " ".join(sys.argv[1:])

print(clf.predict(countVectorizer.transform([query]).todense())[0])