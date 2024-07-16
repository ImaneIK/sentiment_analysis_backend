import re

import nltk
import numpy as np
import matplotlib
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS, cross_origin
import pickle
from collections import Counter
import spacy
from nltk import word_tokenize, PorterStemmer, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from unidecode import unidecode
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textstat.textstat import textstat
import io
import csv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

nlp = spacy.load('fr_core_news_sm')


def download_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')




download_resources()
matplotlib.use('Agg')

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# vectorizer = CountVectorizer(ngram_range=(1, 1))

# Defining additional stopwords
with open('stop_compl.txt') as f:
    stops = [line.strip() for line in f.readlines()]


# Function to predict sentiment probabilities
def predict_sentiment(text):
    cleaned_text = text_clean(text, True)  # Clean the text
    X_test_vect = vectorizer.transform([cleaned_text])
    probabilities = model.predict_proba(X_test_vect)[0]
    prediction = model.predict(X_test_vect)[0]

    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)

    return {
        'prediction': sentiment_labels[prediction],
        'probabilities': probabilities.tolist(),
        'sentiment_scores': sentiment_scores
    }


@app.route('/common_words', methods=['POST'])
def common_words():
    data = request.json
    text = data.get('text', '')

    # Debug: Print original text
    # print("Original Text:", text)

    cleaned_text = clean_text(text)

    # Debug: Print cleaned text
    # print("Cleaned Text:", text)

    # Check if the cleaned text is empty
    if cleaned_text == '':
        return jsonify([])

    vectorizer = CountVectorizer()
    try:
        X = vectorizer.fit_transform([cleaned_text])
    except ValueError as e:
        print("Error:", e)
        return jsonify([])

    # Sum up the counts of each vocabulary word
    word_counts = X.sum(axis=0).A1
    word_freq = [(word, int(word_counts[idx])) for word, idx in
                 vectorizer.vocabulary_.items()]  # Convert to native int
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)[:10]  # Get top 10 words

    return jsonify(word_freq)


def text_statistics(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    num_words = len(words)
    num_sentences = len(sentences)
    num_chars = len(text)
    num_chars_no_spaces = len(text.replace(' ', ''))
    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    most_common_words = Counter(words).most_common(5)
    unique_words = set(words)
    vocabulary_richness = len(unique_words) / num_words if num_words > 0 else 0
    readability_flesch = textstat.flesch_reading_ease(text)
    readability_fk_grade = textstat.flesch_kincaid_grade(text)
    readability_gunning_fog = textstat.gunning_fog(text)

    return {
        'num_words': num_words,
        'num_sentences': num_sentences,
        'num_chars': num_chars,
        'num_chars_no_spaces': num_chars_no_spaces,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'most_common_words': most_common_words,
        'vocabulary_richness': vocabulary_richness,
        'readability_flesch': readability_flesch,
        'readability_fk_grade': readability_fk_grade,
        'readability_gunning_fog': readability_gunning_fog
    }


def split_text_into_sections(text, method='paragraph'):
    if method == 'paragraph':
        sections = text.split('\n\n')  # Divise par paragraphe
    elif method == 'sentence':
        sections = text.split('. ')  # Divise par phrase
    return sections


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sections = sent_tokenize(text)  # Tokenize text into sentences/sections
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    results = []
    for section in sections:
        cleaned_text = clean_text(section)
        vect_section = vectorizer.transform([cleaned_text])
        prediction = model.predict(vect_section)[0]
        # Ensure the prediction is a standard Python type
        prediction = sentiment = sentiment_labels[int(prediction)] if isinstance(prediction, np.integer) else \
            sentiment_labels[prediction]
        results.append({'section': section, 'sentiment': sentiment})

    return jsonify(results)


def extract_named_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entity = ent.text.lower()
        if entity not in entities:
            entities[entity] = {'text': ent.text, 'label': ent.label_}
    return list(entities.values())


@app.route('/text_summary', methods=['POST'])
def text_summary():
    data = request.json
    text = data.get('text', '')
    stats = text_statistics(text)
    return jsonify(stats)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    result = predict_sentiment(text)
    return jsonify(result)


@app.route('/named_entities', methods=['POST'])
def named_entities():
    data = request.json
    text = data.get('text')
    text = preprocess_text(text)
    entities = extract_named_entities(text)
    return jsonify(entities)


@app.route('/generate_word_cloud', methods=['POST'])
def generate_word_cloud_route():
    text = request.json.get('text')
    output_image_path = 'wordcloud.png'
    generate_word_cloud(text, output_image_path)
    return send_file(output_image_path, mimetype='image/png')


# Preprocessing function
def text_clean(text, rm_stop):
    text = re.sub(r'\<.*?\>', '', text)  # Remove HTML tags
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)  # Remove hyperlinks
    text = re.sub(r"\n", "", text)  # Remove line breaks
    text = text.lower()  # Convert to lowercase

    # Removing stop words
    if rm_stop:
        stop_words = set(stopwords.words('french')).union(stops)
        filtered_tokens = [word for word in word_tokenize(text) if word.lower() not in stop_words]
        text = " ".join(filtered_tokens)

    text = re.sub(r"\d+", "", text)  # Remove digits and currencies
    text = re.sub(r'[\$\d+\d+\$]', "", text)
    text = re.sub(r'\d+[\.\/-]\d+[\.\/-]\d+', '', text)  # Remove dates
    text = re.sub(r"\b\w'(\w+)", r'\1', text)  # remove apostrophes
    text = unidecode(text)
    # text = re.sub(r'[^\x00-\x7f]', r' ', text)   # Remove non-ascii
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    porter = PorterStemmer()
    stem_tokens = [porter.stem(word) for word in word_tokenize(text)]
    return " ".join(stem_tokens)

    return text


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r"\b\w'(\w+)", r'\1', text)  # remove apostrophes
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    stop_words = set(stopwords.words('french')).union(stops)
    words = text.split()
    # lemmatizer = WordNetLemmatizer()
    cleaned_text = ' '.join([word for word in words if word not in stop_words])
    return cleaned_text


def generate_circular_mask(diameter):
    x, y = np.ogrid[:diameter, :diameter]
    center = diameter / 2
    mask = (x - center) ** 2 + (y - center) ** 2 > (center ** 2)
    mask = 255 * mask.astype(int)
    return mask


def generate_word_cloud(text, output_image_path):
    text = clean_text(text)

    # Create a circular mask
    diameter = 800
    mask = generate_circular_mask(diameter)

    # Generate the word cloud
    wordcloud = WordCloud(width=diameter, height=diameter, background_color='white', mask=mask).generate(text)

    # Plotting the word cloud
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save the word cloud image
    plt.savefig(output_image_path, format='png')
    plt.close()


def preprocess_text(text):
    # Define the regex pattern to remove leading contractions
    contraction_pattern = re.compile(r"\b(d’|l’|c’|j’|t’|m’|s’|qu’|n’|jusqu’|lorsqu’|puisqu’)(?=\w)")
    # Split text into words and process each word individually
    words = text.split()
    processed_words = [contraction_pattern.sub('', word) for word in words]
    # Rejoin the words into a single string

    # return words
    return ' '.join(processed_words)


@app.route('/export', methods=['POST'])
def export():
    try:
        data = request.get_json()
        articles = data.get('articles', [])

        if not isinstance(articles, list) or not articles:
            return make_response(jsonify({"error": "Invalid articles format"}), 400)

        # Dummy data processing
        article = articles[0]
        title = article.get('title', 'Dummy Title')
        input_article = article.get('input_article', 'Dummy Article')
        publication_date = article.get('publication_date', '2024-01-01')
        prediction = article.get('prediction', 'NaN')
        percentage = article.get('percentage', 'NaN')
        named_entities1 = article.get('named_entities', 'NaN')
        common_words1 = article.get('common_words', 'NaN')

        # Create the CSV file in memory
        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerow(
            ['Title', 'Article', 'Publication Date', 'prediction', 'percentage', 'named entities', 'common words'])
        csv_writer.writerow(
            [title, input_article, publication_date, prediction, percentage, named_entities1, common_words1])
        csv_data = output.getvalue()

        response = make_response(csv_data)
        response.headers['Content-Disposition'] = 'attachment; filename=analysis_results.csv'
        response.headers['Content-type'] = 'text/csv'

        return response
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": str(e)}), 500)


@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.json
        articles = data.get('articles', [])

        if not articles:
            return "No articles provided", 400

        article = articles[0]

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)

        # Set up styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        body_style = styles['BodyText']

        # Create story
        story = []

        # Add title
        title = article.get('title', 'NaN')
        story.append(Paragraph(f"Title: {title}", title_style))

        # Add publication date
        pub_date = article.get('publication_date', 'NaN')
        story.append(Paragraph(f"Publication Date: {pub_date}", body_style))

        # Add text
        input_article = article.get('input_article', 'NaN')
        story.append(Paragraph(f"Article Text: {input_article}", body_style))

        # Add prediction
        prediction = article.get('prediction', 'NaN')
        story.append(Paragraph(f"Sentiment prediction: {prediction}", body_style))

        # Add prediction percentage
        percentage = article.get('percentage', 'NaN')
        story.append(Paragraph(f"Sentiment rate: {percentage}", body_style))

        # named entities
        story.append(Paragraph("\nNamed entities:\n ", body_style))
        named_entities1 = article.get('named_entities', 'NaN')
        for entity in named_entities1:
            entity_text = f"{entity['text']} ({entity['label']})"
            story.append(Paragraph(entity_text))

        # common words
        story.append(Paragraph("\nMost common words:\n ", body_style))
        common_words1 = article.get('common_words', 'NaN')
        for word in common_words1:
            story.append(Paragraph(f"{word} "))

        # Build PDF
        doc.build(story)
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name='article.pdf', mimetype='application/pdf')
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": str(e)}), 500)

# if __name__ == '__main__':
#    app.run(debug=True)
