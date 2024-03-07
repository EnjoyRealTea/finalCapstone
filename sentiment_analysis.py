# =============== Capstone Project - NLP Applications =============
# ************************* E. Thompson ***************************

"""
This program performs sentiment analysis on a sample of a dataset
containing reviews of products from Amazon. The results are examined
by selecting several examples, creating a word cloud, and creating a
stacked bar plot.

Functions
---------
preprocess:
    Takes a string, removes punctuation and stop words, lemmatizes
    and turns into lower case.

predict_sentiment:
    Analyses the sentiment of a string using TextBlob.

polarity_score:
    Returns the polarity of a string, to two decimal places.
"""

from collections import defaultdict
from textblob import TextBlob
from wordcloud import WordCloud
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the en_core_web_sm spaCy model:
nlp = spacy.load('en_core_web_sm')


def preprocess(text):
    """This function cleans a string by removing punctuation and stop
    words, the remaining words are lemmatized and made lower case.
    
    Parameters
    ----------
    text : str
        The text to be cleaned.

    Returns
    -------
    str
        The preprocessed text.
    """
    # Convert text into a SpaCy Doc object:
    doc = nlp(text)

    # Removes stop words, punctuation, lemmatizes and makes lower case:
    clean_text = ' '.join([token.lemma_.lower() for token in doc
                           if not token.is_stop and not token.is_punct])

    return clean_text


def predict_sentiment(text):
    """This function analyses the sentiment of a text string and
    predicts the sentiment, returning the values 'positive', 'negative'
    or 'neutral'.
    
    Parameters
    ----------
    text : str
        The text that the sentiment is predicted for.

    Returns
    -------
    str
        The predicted sentiment: 'positive', 'negative' or 'neutral'.
    """
    # Analyse sentiment using TextBlob:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Convert polarity score into a readable sentiment:
    if polarity > 0:
        sentiment = 'positive'

    elif polarity < 0:
        sentiment = 'negative'

    else:
        sentiment = 'neutral'

    return sentiment


def polarity_score(text):
    """This function returns the strength of the sentiment (polarity)
    found for a text string.

    Parameters
    ----------
    text : str
        The text to be analysed.

    Returns
    -------
    float
        The predicted sentiment strength.
    """
    # Analyse sentiment using TextBlob:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    return round(polarity, 2)


# --- Data preprocessing and sampling ----

# Read the data into a dataframe:
reviews_data = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Drop null values and unneeded columns:
reviews_data = reviews_data[['reviews.text', 'reviews.rating']].dropna()

# Sample the dataset, setting a random state for repeatability:
reviews_sample = reviews_data.sample(100, random_state=11)

# Preprocess the sampled review texts:
reviews_sample['cleaned_reviews'] = reviews_sample['reviews.text'
                                                   ].apply(preprocess)

# --- Performing sentiment analysis on the sample ---

reviews_sample['sentiment'] = reviews_sample['cleaned_reviews'
                                             ].apply(predict_sentiment)

print("--- Sample reviews ---")

# Display two positive reviews from the sample:
print("\nPositive reviews:")
for i in reviews_sample[reviews_sample['sentiment'] == 'positive'
                        ].iloc[0:2].index:
    print(f"""
- {reviews_sample['reviews.text'][i]}
    Rating: {reviews_sample['reviews.rating'][i]}
    Sentiment strength: {polarity_score(reviews_sample['cleaned_reviews'][i])}
    """)

# Display two neutral reviews from the sample:
print("\nNeutral reviews:")
for i in reviews_sample[reviews_sample['sentiment'] == 'neutral'
                        ].iloc[0:2].index:
    print(f"""
- {reviews_sample['reviews.text'][i]}
    Rating: {reviews_sample['reviews.rating'][i]}""")

# Display two negative reviews from the sample:
print("\nNegative reviews:")
for i in reviews_sample[reviews_sample['sentiment'] == 'negative'
                        ].iloc[0:2].index:
    print(f"""
- {reviews_sample['reviews.text'][i]}
    Rating: {reviews_sample['reviews.rating'][i]}
    Sentiment strength: {polarity_score(reviews_sample['cleaned_reviews'][i])}
    """)

# --- Creating a word cloud to analyse word sentiment ---

positive_words = defaultdict(int)
negative_words = defaultdict(int)

for i in reviews_sample.index:
    # Uses preprocessed reviews column:
    words = reviews_sample['cleaned_reviews'][i].split()
    for word in words:
        word_sentiment = polarity_score(word)

        if word_sentiment > 0:
            # Add words with positive sentiment to dictionary:
            positive_words[word] += 1
        elif word_sentiment < 0:
            # Add words with negative sentiment to dictionary:
            negative_words[word] += 1

pos_wordcloud = WordCloud(width=400, height=300, background_color='white'
                          ).generate_from_frequencies(positive_words)
neg_wordcloud = WordCloud(width=400, height=300, background_color='white'
                          ).generate_from_frequencies(negative_words)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(pos_wordcloud, interpolation='bilinear')
ax[0].set_title('Positive Words')
ax[0].axis('off')

ax[1].imshow(neg_wordcloud, interpolation='bilinear')
ax[1].set_title('Negative Words')
ax[1].axis('off')

plt.show()

# --- Plotting and reviewing sample review ratings and sentiments ---

sns.histplot(data=reviews_sample, x='reviews.rating',
             hue='sentiment', multiple="stack", discrete=True)
plt.title("Plot of Review Ratings and their Predicted Sentiment")
plt.xlabel("Review Rating")
plt.ylabel("Number of Reviews")
plt.show()

# Examining reviews in sample that should be negative:
print("\n--- Reviews with a rating of 1.0: ---")
for value in reviews_sample[reviews_sample['reviews.rating'] == 1].values:
    print(f"""\n - {value[0]}
    Sentiment: {value[3]} / {polarity_score(value[2])}""")

# Examining negative reviews from sample with 5.0 rating:
print("\n--- Reviews with negative sentiment and a 5.0 rating: ---")
for value in reviews_sample[(reviews_sample['reviews.rating'] == 5)
                            & (reviews_sample['sentiment'] == 'negative')
                            ].values:
    print(f"""\n - {value[0]}
    Sentiment strength: {polarity_score(value[2])}""")

# --- Comparing the similarity of two reviews ---

# Load medium nlp model in order to use built-in vectorization:
nlp = spacy.load('en_core_web_md')

# Selecting two reviews:
review1 = nlp(reviews_data['reviews.text'][597])
review2 = nlp(reviews_data['reviews.text'][16851])

# Finding the similarity between the reviews:
similarity = review1.similarity(review2)

print(f"""
--- Comparing similarity of two reviews: ---
      
Review 1: '{review1}'
Sentiment: {polarity_score(preprocess(review1))}
Review 2:'{review2}'
Sentiment: {polarity_score(preprocess(review2))}
Similarity: {round(similarity, 3)}.""")
