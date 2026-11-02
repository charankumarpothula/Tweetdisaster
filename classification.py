from google.colab import files
uploaded = files.upload()



import pandas as pd

df = pd.read_csv('tweets.csv')
df.head()



import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (first-time only)
nltk.download('stopwords')




# Define regex patterns
url_pattern = re.compile(r'https?://\S+|www\.\S+')
mention_pattern = re.compile(r'@\w+')
hashtag_pattern = re.compile(r'#(\w+)')
non_alphanum_pattern = re.compile(r'[^a-zA-Z0-9\s]')

# English stopwords list
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    text = str(text).lower()                          # lowercase
    text = url_pattern.sub('', text)                  # remove URLs
    text = mention_pattern.sub('', text)              # remove @mentions
    text = hashtag_pattern.sub(r'\1', text)           # keep hashtag text
    text = non_alphanum_pattern.sub(' ', text)        # remove punctuation/symbols
    text = re.sub(r'\s+', ' ', text).strip()          # remove extra spaces
    # remove stopwords
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)




df['clean_text'] = df['text'].apply(clean_tweet)
df[['text', 'clean_text']].head(10)




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split




X = df['clean_text']      # Features (the cleaned tweet text)
y = df['target']          # Labels (1 = disaster, 0 = not disaster)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)




tfidf = TfidfVectorizer(max_features=5000)  # limit to top 5000 words
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

X_train_tfidf.shape, X_test_tfidf.shape




from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=200)

# Train it
model.fit(X_train_tfidf, y_train)




y_pred = model.predict(X_test_tfidf)




from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))




def predict_tweet(text):
    # Clean the input just like training data
    cleaned = clean_tweet(text)

    # Convert to TF-IDF vector
    vector = tfidf.transform([cleaned])

    # Predict using trained model
    prediction = model.predict(vector)[0]

    # Return readable result
    if prediction == 1:
        return " Real Disaster Tweet"
    else:
        return " Not a Disaster Tweet"



print(predict_tweet("Massive earthquake hits the city, thousands injured."))
print(predict_tweet("I'm so hungry, need a burger right now!"))
print(predict_tweet("Floods destroy homes in my neighborhood "))
print(predict_tweet("What a lovely day to go hiking!"))
