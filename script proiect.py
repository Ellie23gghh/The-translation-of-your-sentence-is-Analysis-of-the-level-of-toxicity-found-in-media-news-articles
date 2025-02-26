import json
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from newsapi import NewsApiClient

# Inițializăm clientul NewsAPI
newsapi = NewsApiClient(api_key='6b2c182698bb406a8c63bb54210a0b77')

# Obținem articolele despre AI
all_articles = newsapi.get_everything(q='ai', language='en')

# Salvăm articolele într-un JSON
with open('articole_ai_noi.json', 'w', encoding='utf-8') as file:
    json.dump(all_articles, file, indent=4)

print("Articles saved to 'articole_ai_noi.json'")

# Încărcăm clasificatorul zero-shot
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Descărcăm stopwords dacă nu sunt deja disponibile
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Curăță textul: elimină caractere speciale și cuvinte de oprire."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Păstrăm doar litere și cifre
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Citim fișierul JSON
with open('articole_ai_noi.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extragem titlurile și descrierile
articles = [
    {
        "title": article.get('title', ''),
        "description": article.get('description', ''),
        "content": article.get('content', '')
    }
    for article in data.get('articles', [])
    if article.get('title') and article.get('description')
]

# Preprocesăm textele
texts = [clean_text(article['title'] + " " + article['description']) for article in articles]

# Definim categoriile pentru clasificare
candidate_labels = ["AI in agriculture", "AI in medicine", "AI in education", "AI in transport", "AI in other domains"]

# Aplicăm clasificarea zero-shot
classified_articles = []
for article, text in zip(articles, texts):
    result = zero_shot_classifier(text, candidate_labels)
    if result['scores'][0] > 0.5:  # Ne asigurăm că predicția are un scor de încredere decent
        label = result['labels'][0]
        score = result['scores'][0]  # Scorul pentru eticheta prezisă
    else:
        label = "Unclassified"
        score = 0.0  # Scor zero pentru etichetele neclasificate

    classified_articles.append({
        "title": article['title'],
        "description": article['description'],
        "category": label,
        "score": score  # Adaugă scorul în rezultatul final
    })

# JSON organizat
with open('articole_clasificate.json', 'w', encoding='utf-8') as file:
    json.dump(classified_articles, file, indent=4)

print("Classified articles saved to 'articole_clasificate.json'")

# Împărțim setul de date pentru evaluare
X_train, X_test, y_train, y_test = train_test_split(texts, [art['category'] for art in classified_articles], test_size=0.2, random_state=42)

# Aplicăm clasificarea pe X_test și salvăm scorurile
predicted_labels = []
predicted_scores = []
for text in X_test:
    result = zero_shot_classifier(text, candidate_labels)
    predicted_labels.append(result['labels'][0] if result['scores'][0] > 0.5 else "Unclassified")
    predicted_scores.append(result['scores'][0] if result['scores'][0] > 0.5 else 0.0)

# Evaluăm acuratețea modelului
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Acuratețea modelului pentru subdomenii este: {accuracy:.2f}")

# Afișăm câteva exemple clasificate, inclusiv scorurile
for text, predicted_label, score in zip(X_test[:5], predicted_labels[:5], predicted_scores[:5]):
    print(f"Text: {text}\nEtichetă prezisă: {predicted_label}\nScor: {score:.2f}\n")
