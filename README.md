# Analysis-of-the-level-of-toxicity-found-in-media-news-articles
Implemented an analysis of the level of toxicity found in media news articles that have AI as their main topic using Python language and Machine Learning algorithms

#Break-down of the script:

1. Fetch AI News Articles
The script uses NewsAPI to fetch AI-related news articles in English and saves them as articole_ai_noi.json.

2. Text Preprocessing
Converts text to lowercase.
Removes special characters.
Eliminates stopwords using NLTK.

3. AI Subdomain Classification
Uses Facebook's BART-large MNLI model for zero-shot classification.
Articles are classified into categories like:AI in Agriculture,AI in Medicine,AI in Education,AI in Transport,AI in Other Domains.
Articles with a classification confidence score below 0.5 are labeled as Unclassified.

4. Saving Results
The classified articles are stored in articole_clasificate.json.

5. Model Evaluation
Splits data into training and testing sets.
Runs classification on test data.
Computes accuracy score.
Displays sample classified articles along with confidence scores.

#Usage:
Run the script using:
python ai_news_toxicity.py

