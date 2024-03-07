# NLP Applications Project: Sentiment Analysis

The program performs sentiment analysis on a sample of a dataset containing reviews of products from Amazon. The results are examined by selecting several examples, creating a word cloud, and creating a stacked bar plot.

The results are discussed and the model evaluated in the pdf file [sentiment_analysis_report.](https://github.com/EnjoyRealTea/finalCapstone/blob/main/sentiment_analysis_report.pdf)

## Contents
  - Installation
  - Useage
    - Examples shown
    - Bar Chart
    - Further examples shown
    - Word Cloud
    - Similarity
  - Credits

## Installation
sentiment_analysis.py can be downloaded and run locally through an IDE using Python.

The following libraries will need to be installed before running:

- pandas
- spacy
- seaborn
- matplotlib
- wordcloud
- textblob

The dataset used is available on Kaggle and can be downloaded here (1491_1.csv):

https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products

>[!IMPORTANT]
>Rename the dataset 'amazon_product_reviews.csv' and place into the same folder as sentiment_analysis.py before running.

## Useage

When run the program will select a sample of 100 rows and preprocess the text of the reviews. It will then analyse the sentiment of the reviews in the sample and display the following so that the performance of the model can be evaluated:

>[!Note]
> To analyse a different sample of reviews, or to change the number of reviews analysed, adjust the sample size and/or random_state on line 122:
> `reviews_sample = reviews_data.sample(100, random_state=11)` 

### 1. A selection of examples
The text, polarity score (sentiment strength) and reviewer rating is shown for:
   - Two reviews identified as having positive sentiment
   - Two reviews identified as having neutral sentiment
   - Two reviews identified as having negative sentiment
     
![Screenshot1](https://github.com/EnjoyRealTea/finalCapstone/assets/153746885/6521f19c-dcc6-44fe-8b2f-cb213949ec03)

### 2. A stacked bar chart
This shows the distribution of ratings and their predicted sentiments. This can be used as a rough guide to how well the model predicts the sentiment of the reviews, and the overall distributions of sentiments.

![ratings_chart1](https://github.com/EnjoyRealTea/finalCapstone/assets/153746885/c70cd25a-bce5-41f9-8623-cd20b030a37a)

### 3. Reviews where the sentiment clearly does not match the reviewer rating
The text and polarity score of reviews with a 1.0 rating and reviews rated as negative but having a 5.0 rating are displayed so that they can be examined to see why the predicted sentiment and reviewer ratings do not match.

![Screenshot2](https://github.com/EnjoyRealTea/finalCapstone/assets/153746885/cc85b6b1-8c2f-4ead-b6dd-5971561b5d6b)
![Screenshot3](https://github.com/EnjoyRealTea/finalCapstone/assets/153746885/a8874f1e-80a3-435d-aed3-abd79b2b09bb)

### 4. A Word Cloud
This displays words regarded as either positive or negative by the model, and their frequencies (indicated by size). This gives further insight into the strengths and limitations of the model as it gives an indication of what types of words are contributing to the polarity scores.
![word_cloud1](https://github.com/EnjoyRealTea/finalCapstone/assets/153746885/eb63219d-fe9e-459d-85da-54db9a2a40f4)
### 5. A comparison of two reviews
The similarity of two reviews is shown to give an indication of the variability of the reviews in the dataset.
![Screenshot4](https://github.com/EnjoyRealTea/finalCapstone/assets/153746885/cacdb634-6c7c-4478-80ed-1db59c0af25f)

>[!Note]
>For a discussion of the results and how they relate to the performance of the model, please see the [sentiment_analysis_report.](https://github.com/EnjoyRealTea/finalCapstone/blob/main/sentiment_analysis_report.pdf) 

## Credits

sentiment_analysis.py and sentiment_analysis_report.pdf were written by E. Thompson to submit as a final Capstone Project for the Hyperion Dev / CoGrammar Data Science skills bootcamp.


The dataset originates from Datafiniti's Product Database and can be found [here.](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)



