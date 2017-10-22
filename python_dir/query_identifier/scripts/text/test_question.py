import sys
import pickle

countVectorizer = pickle.load(open("vectorizer_question", "rb"))
clf = pickle.load(open("clf_question", "rb"))

query = " ".join(sys.argv[1:])

print(clf.predict(countVectorizer.transform([query]).todense())[0])
