# Project-Ecommerce-Reviews---Sentiment-Analysis

# Sentiment Analysis on Product Reviews

Sentiment classification (positive/negative) on e-commerce reviews, comparing
lexicon-based approaches (spaCy + LeIA/LeIAMBA) with supervised models (Naive Bayes,
Logistic Regression, SVM).

## Context

Sentiment labels were derived from review ratings (0-5 stars):
- Scores 0-2 → negative
- Scores 4-5 → positive
- Score 3 → initially treated as "neutral"

## Key decisions and findings

### The "neutral" class problem

A manual inspection of score-3 reviews revealed that most of them actually expressed
**negative** sentiment in the text (complaints about delivery time, size, quality),
not neutrality. This made the "neutral" class noisy and hurt every model's
performance on that category.

**Decision:** reframe the problem as binary classification (positive/negative), since
the "neutral" label based solely on the rating didn't reflect the actual content of
the text.

### Class balancing

The dataset is imbalanced (positive >> negative). The following were tested:
- `class_weight='balanced'`
- Manual weights (e.g. `{"positive": 1, "negative": 2}`)
- `sample_weight` for models without native `class_weight` support (Naive Bayes)

## Pipeline

1. Text preprocessing (lemmatization with spaCy, stopword/punctuation removal)
2. TF-IDF vectorization
3. Train/test split (80/20, stratified by class)
4. Model training and comparison
5. Evaluation with classification_report and confusion matrix

## Technologies

- scikit-learn
- spaCy
- LeIA / LeIAMBA (Portuguese sentiment lexicon)
- pandas

