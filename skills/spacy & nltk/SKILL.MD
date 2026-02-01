---
name: spacy-nltk
description: Natural Language Processing for text analysis, corpus linguistics, and production NLP pipelines. spaCy provides fast production-grade tokenization, POS tagging, NER, dependency parsing, and custom model training. NLTK provides classical corpus linguistics, linguistic analysis, VADER sentiment, collocation analysis, and access to standard linguistic corpora. Use when: processing and analyzing text data, extracting named entities (people, orgs, locations, dates), dependency parsing and syntactic analysis, building text classification pipelines, performing corpus-level linguistic analysis (frequency, collocations, readability), sentiment analysis, lemmatization and stemming, working with multilingual text, training custom NER or text classifiers, or any task requiring structured understanding of natural language beyond simple string operations.
version: 3.7.0
license: MIT (spaCy), BSD (NLTK)
---

# spaCy & NLTK — Natural Language Processing

Two complementary libraries that together cover the full NLP stack. **spaCy** is the production engine: fast, opinionated, pipeline-based. **NLTK** is the linguistics toolkit: deep, flexible, corpus-rich. Use them together — spaCy for processing, NLTK for analysis.

## Library Decision Matrix

```
USE spaCy WHEN:                         USE NLTK WHEN:
─────────────────────────────           ─────────────────────────────
Processing speed matters                Linguistic analysis depth matters
Named Entity Recognition (NER)          Corpus linguistics (collocations, freq)
Dependency parsing                      Working with standard corpora (Brown, Penn)
Production pipelines                    VADER sentiment analysis
Custom model training                   Stemming (Porter, Snowball)
Multilingual text                       POS tag evaluation against gold sets
Document/entity similarity              Readability metrics
Pipeline: tokenize→tag→parse→NER       Classical grammar analysis
Embedding-based operations              Educational / research NLP
```

## Reference Documentation

**spaCy docs**: https://spacy.io/usage  
**spaCy API**: https://spacy.io/api  
**NLTK docs**: https://www.nltk.org/  
**NLTK book**: https://www.nltk.org/book/  
**Search patterns**: `spacy.load`, `nlp(text)`, `doc.ents`, `nltk.word_tokenize`, `FreqDist`

## Core Principles

### spaCy: Pipeline Architecture
`spacy.load()` returns an `nlp` object — a processing pipeline. Calling `nlp(text)` runs the full pipeline (tokenizer → tagger → parser → NER) and returns a `Doc`. Everything — tokens, sentences, entities, syntax — is extracted in a single pass. This is the key performance advantage.

### NLTK: Function-Based Analysis
NLTK works at the function level. You call individual functions: `word_tokenize()`, `pos_tag()`, `FreqDist()`. No shared state or pipeline. More granular control; less automatic.

### Linguistic Hierarchy
Text → Sentences → Tokens → Morphemes. Both libraries operate at these levels but with different strengths: spaCy owns the token→entity→syntax layer; NLTK owns the corpus→frequency→collocation layer.

### spaCy Models Are Language-Specific
`en_core_web_sm` is English small. `de_core_news_md` is German medium. Model size affects accuracy: sm (speed) → md (balanced) → lg (accuracy + word vectors). Download before use: `python -m spacy download en_core_web_sm`.

## Quick Reference

### Installation

```bash
pip install spacy nltk

# Download spaCy models
python -m spacy download en_core_web_sm     # English small — fast, most tasks
python -m spacy download en_core_web_md     # English medium — better accuracy + vectors
python -m spacy download en_core_web_lg     # English large — best accuracy + full vectors

# NLTK data (run once in Python)
import nltk
nltk.download('punkt_tab')        # Tokenizer
nltk.download('averaged_perceptron_tagger_eng')  # POS tagger
nltk.download('stopwords')        # Stop words
nltk.download('vader_lexicon')    # Sentiment
nltk.download('brown')            # Brown corpus
nltk.download('wordnet')          # WordNet lexical database
```

### Standard Imports

```python
import spacy
import nltk
from collections import Counter
import pandas as pd
```

### Basic Pattern — spaCy Full Pipeline

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is headquartered in Cupertino, California. Tim Cook is the CEO.")

# Everything extracted in one pass:
for token in doc:
    print(f"{token.text:15s} POS={token.pos_:6s} lemma={token.lemma_:15s} dep={token.dep_}")

print("\nEntities:")
for ent in doc.ents:
    print(f"  {ent.text:20s} → {ent.label_}")

print("\nSentences:")
for sent in doc.sents:
    print(f"  {sent.text}")
```

### Basic Pattern — NLTK Linguistic Analysis

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from nltk.corpus import stopwords

text = "Apple is headquartered in Cupertino, California. Tim Cook is the CEO."

# Tokenize and tag
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
print("Tagged:", tagged)

# Frequency analysis
words   = [t for t in tokens if t.isalpha()]
stops   = set(stopwords.words('english'))
cleaned = [w.lower() for w in words if w.lower() not in stops]
freq    = FreqDist(cleaned)
freq.most_common(10)
```

## Critical Rules

### ✅ DO

- **Load spaCy model ONCE, reuse `nlp` object** — `spacy.load()` is expensive. Load at module level, call `nlp()` per document.
- **Use `doc.ents` for NER, never regex first** — spaCy's statistical NER outperforms regex for ambiguous entities. Use regex only as fallback for structured patterns (emails, phone numbers).
- **Use `token.lemma_` over `token.text`** — Lemmatization normalizes "running"→"run", "better"→"good". Essential before frequency analysis.
- **Use `nlp.pipe()` for batch processing** — Processes documents in parallel. 10–100x faster than looping `nlp()` per document.
- **Disable unused pipeline components** — `nlp.select_pipes(disable=['parser'])` if you only need NER. Significant speedup.
- **Use `token.is_stop`, `token.is_punct`, `token.is_alpha`** — spaCy's built-in filters. Cleaner than manual regex.
- **Use NLTK's `FreqDist` and collocations for corpus-level analysis** — Built for this exact purpose.
- **Prefer `md` or `lg` models for similarity tasks** — `sm` models have no word vectors; `.similarity()` will fail or return meaningless scores.

### ❌ DON'T

- **Don't reload `nlp` inside loops** — Catastrophic performance. Load once.
- **Don't use spaCy `sm` models for `.similarity()`** — No word vectors in small models. Returns 0 or garbage.
- **Don't mix spaCy and NLTK tokenizers on the same text** — Different tokenization rules produce different token boundaries. Pick one per pipeline.
- **Don't assume `doc.ents` covers everything** — NER recall is ~80–90%. Post-process with rules for domain-specific entities.
- **Don't use `nltk.word_tokenize` in production loops** — Slow per-call overhead. Use spaCy for throughput.
- **Don't skip model validation** — Always check entity/POS accuracy on a sample of your domain before trusting at scale.

## Anti-Patterns (NEVER)

```python
import spacy

# ❌ BAD: Loading model inside a loop — 5–10 seconds per load
results = []
for text in documents:
    nlp = spacy.load('en_core_web_sm')   # Reloaded 10,000 times
    doc = nlp(text)
    results.append(doc.ents)

# ✅ GOOD: Load once, process all
nlp     = spacy.load('en_core_web_sm')
results = [list(nlp(text).ents) for text in documents]

# ─────────────────────────────────────────────────────────

# ❌ BAD: Similarity on small model — no vectors, meaningless result
nlp = spacy.load('en_core_web_sm')          # sm = no word vectors
doc1 = nlp("machine learning")
doc2 = nlp("deep learning")
print(doc1.similarity(doc2))                 # Returns ~0.0 or warning

# ✅ GOOD: Use md or lg which include word vectors
nlp = spacy.load('en_core_web_md')
doc1 = nlp("machine learning")
doc2 = nlp("deep learning")
print(doc1.similarity(doc2))                 # Returns 0.85 — meaningful

# ─────────────────────────────────────────────────────────

# ❌ BAD: Sequential processing — slow for thousands of docs
results = []
for text in documents:
    doc = nlp(text)
    results.append(doc)

# ✅ GOOD: nlp.pipe() — parallel, batched, 10–100x faster
results = list(nlp.pipe(documents, batch_size=64, n_process=4))

# ─────────────────────────────────────────────────────────

# ❌ BAD: Regex NER when spaCy already does this
import re
entities = re.findall(r'\b[A-Z][a-z]+\b', text)   # Catches "The", "California", noise

# ✅ GOOD: Statistical NER — context-aware, handles ambiguity
doc       = nlp(text)
entities  = [(ent.text, ent.label_) for ent in doc.ents]
```

## spaCy — Core Operations

### Token Properties

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("The quick brown fox jumped over 3 lazy dogs in 2024.")

for token in doc:
    print(
        f"text={token.text:12s} "
        f"lemma={token.lemma_:12s} "
        f"pos={token.pos_:6s} "       # Coarse POS: NOUN, VERB, ADJ...
        f"tag={token.tag_:6s} "       # Fine POS: NN, NNS, VBD...
        f"dep={token.dep_:8s} "       # Syntactic role: nsubj, dobj...
        f"stop={token.is_stop} "
        f"punct={token.is_punct} "
        f"alpha={token.is_alpha} "
        f"num={token.is_digit}"
    )

# Key properties reference:
# token.text          → original text
# token.lemma_        → base form ("running" → "run")
# token.pos_          → coarse POS tag
# token.tag_          → fine-grained POS tag
# token.dep_          → dependency relation
# token.head          → syntactic head token
# token.children      → syntactic dependents (generator)
# token.is_stop       → is stop word?
# token.is_punct      → is punctuation?
# token.is_alpha      → all alphabetic?
# token.i             → index in doc
# token.idx           → character offset in original text
# token.sent          → sentence containing this token
```

### Named Entity Recognition

```python
import spacy

nlp = spacy.load('en_core_web_sm')
text = """
Google acquired Waze in 2013 for approximately $1.15 billion.
The deal was announced by Sundar Pichai in Mountain View, California.
"""
doc = nlp(text)

# All entities
for ent in doc.ents:
    print(f"{ent.text:25s} → {ent.label_:10s} (start={ent.start_char}, end={ent.end_char})")

# Entity labels reference:
# PERSON   — people names
# ORG      — companies, agencies, institutions
# GPE      — geopolitical entities (countries, cities, states)
# LOC      — non-GPE locations (mountains, rivers)
# DATE     — dates or periods
# MONEY    — monetary values
# PRODUCT  — products
# EVENT    — named events
# WORK_OF_ART — titles of books, songs, etc.
# NORP     — nationalities, religious, political groups

# Filter by entity type
people = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
orgs   = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
money  = [ent.text for ent in doc.ents if ent.label_ == 'MONEY']

# Entity spans (for surrounding context)
for ent in doc.ents:
    start = max(0, ent.start - 3)
    end   = min(len(doc), ent.end + 3)
    context = doc[start:end].text
    print(f"{ent.label_:10s} | {ent.text:20s} | Context: \"{context}\"")
```

### Dependency Parsing

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("The CEO of Apple announced a new product.")

# Tree structure — find the root and walk the tree
for token in doc:
    print(f"{token.text:12s} ← dep={token.dep_:10s} ← head={token.head.text}")

# Extract subject-verb-object triples
def extract_svo(doc):
    """Extract (subject, verb, object) triples from dependency parse."""
    triples = []
    for token in doc:
        if token.pos_ == 'VERB':
            subj = [t.text for t in token.children if t.dep_ in ('nsubj', 'nsubjpass')]
            obj  = [t.text for t in token.children if t.dep_ in ('dobj', 'attr', 'pobj')]
            if subj and obj:
                triples.append((subj[0], token.lemma_, obj[0]))
    return triples

doc = nlp("John likes pizza. Mary wrote a paper. The company acquired a startup.")
for s, v, o in extract_svo(doc):
    print(f"  {s} —[{v}]→ {o}")
# John —[like]→ pizza
# Mary —[write]→ paper
# company —[acquire]→ startup
```

### Batch Processing and Performance

```python
import spacy
import time

nlp = spacy.load('en_core_web_sm')

# ─── nlp.pipe() — the standard batch API ───
documents = ["Text one.", "Text two.", "Text three."] * 1000  # 3000 docs

# Parallel processing with batching
docs = list(nlp.pipe(documents, batch_size=64, n_process=4))

# ─── Disable unused components for speed ───
# If you only need tokenization + NER, skip the parser:
with nlp.select_pipes(disable=['parser']):
    docs = list(nlp.pipe(documents, batch_size=128))

# Available components to selectively disable:
# 'tagger'   — POS tagging
# 'parser'   — dependency parsing (also controls sentence segmentation)
# 'ner'      — named entity recognition
# 'lemmatizer' — lemmatization

# ─── Processing stats ───
start = time.time()
docs  = list(nlp.pipe(documents, batch_size=64))
elapsed = time.time() - start
print(f"Processed {len(documents)} docs in {elapsed:.2f}s "
      f"({len(documents)/elapsed:.0f} docs/sec)")
```

## NLTK — Corpus Linguistics

### Corpus Analysis

```python
import nltk
from nltk.corpus import brown, gutenberg
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# ─── Brown corpus: tagged by genre ───
# Genres: adventure, fiction, hobbies, humor, learned, lore,
#         mystery, news, romance, science_fiction, skill_trade, social

# Word frequency by genre
stops = set(stopwords.words('english'))

genre_freq = {}
for genre in brown.fileids()[:5]:                    # Sample 5 genres
    words = [w.lower() for w in brown.words(genre) if w.isalpha() and w.lower() not in stops]
    genre_freq[genre] = FreqDist(words)

# Top 10 words per genre
for genre, freq in genre_freq.items():
    print(f"\n{genre}:")
    print(f"  {freq.most_common(10)}")

# ─── Gutenberg corpus: classic literature ───
for name in gutenberg.fileids():
    words = gutenberg.words(name)
    sents = gutenberg.sents(name)
    print(f"{name:35s} words={len(words):>8,}  sents={len(sents):>5,}  "
          f"avg_len={len(words)/len(sents):.1f}")
```

### Collocation Analysis

```python
import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures

# Text as word list
text = """Machine learning is a subset of artificial intelligence that gives systems
the ability to learn from data. Deep learning is a subset of machine learning
that uses neural networks."""
words = nltk.word_tokenize(text.lower())

# Bigram collocations — phrases that co-occur more than chance
bi_finder  = BigramCollocationFinder.from_words(words)
bi_colocs  = bi_finder.nbest(BigramAssocMeasures.pmi, 10)   # PMI = Pointwise Mutual Information
print("Top bigram collocations (PMI):")
for pair in bi_colocs:
    print(f"  {pair[0]:15s} {pair[1]}")

# Trigram collocations
tri_finder = TrigramCollocationFinder.from_words(words)
tri_colocs = tri_finder.nbest(TrigramAssocMeasures.pmi, 5)
print("\nTop trigram collocations:")
for triple in tri_colocs:
    print(f"  {' '.join(triple)}")

# ─── On larger corpus ───
from nltk.corpus import brown
corpus_words = [w.lower() for w in brown.words() if w.isalpha()]
finder = BigramCollocationFinder.from_words(corpus_words)

# Filter by minimum frequency, then rank by association
finder.apply_freq_filter(5)            # Ignore pairs appearing < 5 times
finder.apply_word_filter(lambda w: len(w) < 3)  # Ignore words < 3 chars
top_colocs = finder.nbest(BigramAssocMeasures.pmi, 20)
```

### VADER Sentiment Analysis

```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# ─── Single text ───
text = "I absolutely love this product! It's the best thing I've ever bought."
scores = sia.polarity_scores(text)
print(scores)
# {'neg': 0.0, 'neu': 0.346, 'pos': 0.654, 'compound': 0.8802}

# compound score: -1.0 (most negative) → +1.0 (most positive)
# Convention: compound >= 0.05  → positive
#             compound <= -0.05 → negative
#             else              → neutral

def classify_sentiment(text):
    compound = sia.polarity_scores(text)['compound']
    if   compound >= 0.05:  return 'positive'
    elif compound <= -0.05: return 'negative'
    else:                   return 'neutral'

# ─── Batch classification ───
import pandas as pd

texts = [
    "This movie was fantastic!",
    "Terrible experience, never again.",
    "The weather is okay today.",
    "I'm so disappointed with the service.",
    "Amazing food and great atmosphere!"
]

results = pd.DataFrame({
    'text':      texts,
    'scores':    [sia.polarity_scores(t) for t in texts],
    'sentiment': [classify_sentiment(t) for t in texts]
})
results[['compound', 'pos', 'neg', 'neu']] = pd.json_normalize(results['scores'])
print(results[['text', 'sentiment', 'compound']])
```

### WordNet — Lexical Database

```python
import nltk
from nltk.corpus import wordnet as wn

# ─── Synsets: meaning clusters ───
# "bank" has multiple meanings
for synset in wn.synsets('bank'):
    print(f"  {synset.name():25s} → {synset.definition()}")
# bank.n.01 → a financial institution...
# bank.n.02 → sloping land...
# bank.v.01 → to put in a bank...

# ─── Hypernyms (is-a relationship) — walk up the taxonomy ───
dog = wn.synset('dog.n.01')
print("Dog hypernyms:", [h.name() for h in dog.hypernyms()])
# → ['carnivore.n.01']

# ─── Hyponyms (subtypes) ───
animal = wn.synset('animal.n.01')
print("Animal types:", [h.name() for h in animal.hyponyms()][:10])

# ─── Similarity between concepts ───
cat    = wn.synset('cat.n.01')
dog    = wn.synset('dog.n.01')
car    = wn.synset('car.n.01')

print(f"cat ↔ dog:  {cat.wup_similarity(dog):.2f}")   # Semantic similarity 0–1
print(f"cat ↔ car:  {cat.wup_similarity(car):.2f}")   # Should be low
```

## Text Preprocessing Pipeline

```python
import spacy
import re
import pandas as pd

nlp = spacy.load('en_core_web_sm')

def preprocess(text: str,
               lowercase: bool = True,
               remove_punct: bool = True,
               remove_stops: bool = True,
               lemmatize: bool = True,
               remove_numbers: bool = True) -> list[str]:
    """
    Full preprocessing pipeline using spaCy.
    Returns list of cleaned tokens.
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        if remove_punct and token.is_punct: continue
        if remove_stops and token.is_stop:  continue
        if remove_numbers and token.like_num: continue
        if not token.is_alpha:              continue

        word = token.lemma_ if lemmatize else token.text
        word = word.lower() if lowercase else word
        tokens.append(word)

    return tokens

# Example
text = "The 3 Quick Brown Foxes jumped over 12 lazy dogs in New York!"
print(preprocess(text))
# ['quick', 'brown', 'fox', 'jump', 'lazy', 'dog', 'new', 'york']

# ─── Batch preprocessing for a DataFrame ───
df = pd.DataFrame({'text': [
    "Running fast is great exercise!",
    "The 100 runners finished in 2 hours.",
    "Machine learning models are powerful tools."
]})
df['tokens'] = df['text'].apply(preprocess)
df['token_str'] = df['tokens'].apply(lambda t: ' '.join(t))
print(df[['text', 'token_str']])
```

## Text Classification Pipeline

```python
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text)
    return ' '.join([t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop])

# ─── Sample data ───
texts  = ["I love this product"] * 50 + ["terrible, worst ever"] * 50 + ["it was okay"] * 50
labels = ['pos'] * 50 + ['neg'] * 50 + ['neu'] * 50

# Preprocess
texts_clean = [preprocess(t) for t in texts]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels, test_size=0.3, random_state=42, stratify=labels
)

# TF-IDF features + Logistic Regression
tfidf      = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_v  = tfidf.fit_transform(X_train)
X_test_v   = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_v, y_train)

y_pred = clf.predict(X_test_v)
print(classification_report(y_test, y_pred))

# ─── Predict new text ───
new_texts = ["This is amazing!", "Awful experience.", "It was fine."]
new_clean = [preprocess(t) for t in new_texts]
new_vec   = tfidf.transform(new_clean)
preds     = clf.predict(new_vec)
probs     = clf.predict_proba(new_vec)

for text, pred, prob in zip(new_texts, preds, probs):
    print(f"{text:30s} → {pred:8s} (confidence: {max(prob):.2f})")
```

## spaCy Custom NER Training

```python
import spacy
from spacy.tokens import Doc
import random
import json

# ─── Training data format ───
# Each example: (text, {"entities": [(start_char, end_char, label)]})
TRAIN_DATA = [
    ("Apple reported $394B in revenue.", {"entities": [(0, 5, "COMPANY"), (17, 23, "REVENUE")]}),
    ("Google acquired Waze for $1.15B.", {"entities": [(0, 6, "COMPANY"), (16, 20, "COMPANY"), (25, 31, "REVENUE")]}),
    ("Microsoft invested $10B in OpenAI.", {"entities": [(0, 9, "COMPANY"), (20, 24, "REVENUE"), (28, 34, "COMPANY")]}),
]

# ─── Training pipeline ───
def train_custom_ner(train_data, model_name='en_core_web_sm', n_iter=30, output_path='custom_ner'):
    nlp = spacy.load(model_name)

    # If NER component doesn't exist, add it
    if 'ner' not in nlp.pipe_names:
        ner = spacy.pipeline.entity_ruler(nlp)
    else:
        ner = nlp.get_pipe('ner')

    # Add entity labels
    labels = set()
    for _, ents in train_data:
        for _, _, label in ents['entities']:
            labels.add(label)
    for label in labels:
        ner.add_label(label)

    # Freeze other pipeline components — only train NER
    unaffected_pipes = [name for name in nlp.pipe_names if name != 'ner']

    with nlp.select_pipes(disable=unaffected_pipes):
        optimizer = nlp.begin_training()
        for i in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                doc  = nlp.make_doc(text)
                example = spacy.tokens.Doc(nlp.vocab, words=[t.text for t in doc])
                losses.update(nlp.update([doc], [annotations], losses=losses, sgd=optimizer))
            print(f"Epoch {i+1:3d}: losses = {losses}")

    # Save
    nlp.to_disk(output_path)
    print(f"Model saved to {output_path}/")
    return nlp

# ─── Usage ───
# nlp = train_custom_ner(TRAIN_DATA)
# doc = nlp("Tesla announced $80B in sales for the year.")
# for ent in doc.ents:
#     print(ent.text, ent.label_)
```

## Word and Document Similarity

```python
import spacy
import numpy as np

nlp = spacy.load('en_core_web_md')    # md required for word vectors

# ─── Token-level similarity ───
doc = nlp("machine learning deep learning artificial intelligence")
tokens = list(doc)

# Pairwise similarity matrix
import pandas as pd

words = ['machine', 'learning', 'deep', 'artificial', 'intelligence']
docs  = [nlp(w) for w in words]
sim_matrix = np.array([[d1.similarity(d2) for d2 in docs] for d1 in docs])

df_sim = pd.DataFrame(sim_matrix, index=words, columns=words)
print(df_sim.round(2))

# ─── Document-level similarity ───
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks for pattern recognition.",
    "The stock market crashed yesterday afternoon.",
    "Financial markets experienced significant volatility.",
    "Cats and dogs are popular household pets."
]

doc_objs = [nlp(text) for text in documents]

print("\nDocument similarity matrix:")
for i, d1 in enumerate(doc_objs):
    for j, d2 in enumerate(doc_objs):
        if i < j:
            sim = d1.similarity(d2)
            if sim > 0.7:
                print(f"  [{sim:.2f}] \"{documents[i][:40]}...\" ↔ \"{documents[j][:40]}...\"")

# ─── Find most similar document to a query ───
query = nlp("How do neural networks learn?")
scored = [(doc, text, query.similarity(doc)) for doc, text in zip(doc_objs, documents)]
scored.sort(key=lambda x: x[2], reverse=True)

print(f"\nQuery: \"{query.text}\"")
for _, text, score in scored[:3]:
    print(f"  [{score:.2f}] {text}")
```

## Practical Workflows

### 1. Entity-Centric Document Analysis

```python
import spacy
import pandas as pd
from collections import defaultdict, Counter

nlp = spacy.load('en_core_web_sm')

def analyze_entities(texts: list[str]) -> dict:
    """
    Full entity analysis across a corpus:
    → entity counts, co-occurrences, per-document entity sets.
    """
    entity_counter    = Counter()
    entity_by_type    = defaultdict(Counter)
    cooccurrence      = Counter()
    doc_entities      = []

    docs = list(nlp.pipe(texts, batch_size=64))

    for doc in docs:
        doc_ents = [(ent.text, ent.label_) for ent in doc.ents]
        doc_entities.append(doc_ents)

        # Counts
        for text, label in doc_ents:
            entity_counter[text] += 1
            entity_by_type[label][text] += 1

        # Co-occurrence: which entities appear in the same document
        unique_ents = list(set(e[0] for e in doc_ents))
        for i in range(len(unique_ents)):
            for j in range(i + 1, len(unique_ents)):
                pair = tuple(sorted([unique_ents[i], unique_ents[j]]))
                cooccurrence[pair] += 1

    return {
        'entity_counts':       entity_counter,
        'entities_by_type':    dict(entity_by_type),
        'cooccurrence':        cooccurrence,
        'doc_entities':        doc_entities
    }

def print_report(results):
    print("=== TOP ENTITIES ===")
    for ent, count in results['entity_counts'].most_common(10):
        print(f"  {ent:25s} {count:>5,}")

    print("\n=== BY TYPE ===")
    for etype, counter in results['entities_by_type'].items():
        print(f"\n  {etype}:")
        for ent, count in counter.most_common(5):
            print(f"    {ent:25s} {count:>5,}")

    print("\n=== TOP CO-OCCURRENCES ===")
    for (e1, e2), count in results['cooccurrence'].most_common(10):
        print(f"  {e1:20s} ↔ {e2:20s}  ({count}×)")

# ─── Usage ───
# texts  = load_news_articles()
# results = analyze_entities(texts)
# print_report(results)
```

### 2. Corpus Linguistics Report

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.corpus import stopwords
import pandas as pd
import math

def corpus_report(texts: list[str], title: str = "Corpus") -> pd.DataFrame:
    """
    Comprehensive corpus linguistics analysis:
    → vocabulary richness, frequency distribution,
      collocations, readability estimate.
    """
    stops     = set(stopwords.words('english'))
    all_words = []
    all_sents = []

    for text in texts:
        tokens = word_tokenize(text)
        all_words.extend(tokens)
        # Simple sentence split on period
        all_sents.extend([s.strip() for s in text.split('.') if s.strip()])

    # Clean tokens
    clean_words = [w.lower() for w in all_words if w.isalpha() and w.lower() not in stops]
    total_words = len(all_words)
    total_sents = len(all_sents)
    unique_words = len(set(w.lower() for w in all_words if w.isalpha()))

    # Frequency distribution
    freq = FreqDist(clean_words)

    # Type-Token Ratio: vocabulary richness (0–1, higher = richer)
    ttr = unique_words / total_words if total_words > 0 else 0

    # Average sentence length
    avg_sent_len = total_words / total_sents if total_sents > 0 else 0

    # Lexical diversity (log-corrected TTR — comparable across corpus sizes)
    # CTTR = unique / sqrt(2 * total)
    cttr = unique_words / math.sqrt(2 * total_words) if total_words > 0 else 0

    # Collocations
    finder  = BigramCollocationFinder.from_words(clean_words)
    finder.apply_freq_filter(3)
    colocs  = finder.nbest(BigramAssocMeasures.pmi, 10)

    # Build report
    print(f"\n{'='*50}")
    print(f"  CORPUS REPORT: {title}")
    print(f"{'='*50}")
    print(f"  Total words:          {total_words:>10,}")
    print(f"  Total sentences:      {total_sents:>10,}")
    print(f"  Unique words:         {unique_words:>10,}")
    print(f"  Type-Token Ratio:     {ttr:>10.3f}  (0–1, higher = richer)")
    print(f"  Log-corrected TTR:    {cttr:>10.3f}  (size-independent)")
    print(f"  Avg sentence length:  {avg_sent_len:>10.1f} words")

    print(f"\n  TOP 15 WORDS (after stopword removal):")
    for word, count in freq.most_common(15):
        bar = '█' * int(count / freq.most_common(1)[0][1] * 30)
        print(f"    {word:20s} {count:>6,}  {bar}")

    print(f"\n  TOP COLLOCATIONS (PMI):")
    for pair in colocs:
        print(f"    {pair[0]} {pair[1]}")

    return freq

# ─── Usage ───
# texts  = open('corpus.txt').read().split('\n\n')
# freq = corpus_report(texts, title="News Articles 2024")
```

### 3. Multi-Stage NLP Pipeline

```python
import spacy
import pandas as pd
from collections import Counter

nlp = spacy.load('en_core_web_sm')

def full_nlp_pipeline(texts: list[str]) -> pd.DataFrame:
    """
    Single pass through corpus:
    tokenize → POS tag → NER → sentence split → sentiment (VADER) → extract features.
    Returns one row per document with all extracted features.
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    rows = []
    docs = list(nlp.pipe(texts, batch_size=64))

    for text, doc in zip(texts, docs):
        # Entity extraction
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        ent_types = Counter(label for _, label in entities)

        # POS distribution
        pos_counts = Counter(token.pos_ for token in doc)

        # Sentence count
        sents = list(doc.sents)

        # Sentiment (VADER)
        sentiment = sia.polarity_scores(text)

        # Token-level features
        tokens       = [t for t in doc if t.is_alpha and not t.is_stop and not t.is_punct]
        unique_lemmas = set(t.lemma_.lower() for t in tokens)

        rows.append({
            'text':              text,
            'n_tokens':          len(doc),
            'n_sentences':       len(sents),
            'n_entities':        len(entities),
            'entities':          entities,
            'n_unique_lemmas':   len(unique_lemmas),
            'lex_diversity':     len(unique_lemmas) / len(tokens) if tokens else 0,
            'sentiment_compound': sentiment['compound'],
            'sentiment_label':   ('positive' if sentiment['compound'] >= 0.05
                                  else 'negative' if sentiment['compound'] <= -0.05 else 'neutral'),
            'pos_noun':          pos_counts.get('NOUN', 0),
            'pos_verb':          pos_counts.get('VERB', 0),
            'pos_adj':           pos_counts.get('ADJ', 0),
            'ent_person':        ent_types.get('PERSON', 0),
            'ent_org':           ent_types.get('ORG', 0),
            'ent_gpe':           ent_types.get('GPE', 0),
        })

    return pd.DataFrame(rows)

# ─── Usage ───
# texts = ["Google hired 5000 engineers in San Francisco.",
#          "The stock market crashed, sending investors into panic.",
#          "Tim Cook announced a beautiful new product at WWDC."]
# df = full_nlp_pipeline(texts)
# print(df[['text', 'sentiment_label', 'n_entities', 'lex_diversity']].to_string())
```

### 4. Keyword and Keyphrase Extraction

```python
import spacy
from collections import Counter
import math

nlp = spacy.load('en_core_web_sm')

def extract_keywords(texts: list[str], top_n: int = 20) -> list[tuple[str, float]]:
    """
    TF-IDF keyword extraction using spaCy lemmas.
    Returns terms with highest TF-IDF scores across the corpus.
    """
    # Tokenize all docs: lemmatize, remove stops/punct/numbers
    doc_terms = []
    for text in texts:
        doc   = nlp(text)
        terms = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and not t.is_punct]
        doc_terms.append(terms)

    n_docs = len(doc_terms)

    # Document frequency
    df = Counter()
    for terms in doc_terms:
        for term in set(terms):
            df[term] += 1

    # TF-IDF per document, then average across corpus
    tfidf_sum = Counter()

    for terms in doc_terms:
        tf = Counter(terms)
        doc_len = len(terms) if terms else 1
        for term, count in tf.items():
            tf_score    = count / doc_len
            idf_score   = math.log(n_docs / (1 + df[term]))
            tfidf_sum[term] += tf_score * idf_score

    # Normalize by number of documents containing the term
    tfidf_avg = {term: score / df[term] for term, score in tfidf_sum.items() if df[term] >= 2}

    # Sort by score
    ranked = sorted(tfidf_avg.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# ─── Usage ───
# texts = load_article_corpus()
# keywords = extract_keywords(texts, top_n=15)
# for kw, score in keywords:
#     print(f"  {kw:25s} {score:.4f}")
```

## Performance and Scaling

### Processing Speed Benchmarks

```python
import spacy
import time

nlp = spacy.load('en_core_web_sm')

# Scenario A: 10k documents, full pipeline
docs_10k = ["Short sentence with entities like Google and Apple."] * 10000

start = time.time()
processed = list(nlp.pipe(docs_10k, batch_size=128, n_process=4))
elapsed   = time.time() - start
print(f"Full pipeline: {len(docs_10k)} docs in {elapsed:.1f}s ({len(docs_10k)/elapsed:.0f} docs/s)")

# Scenario B: Same with parser disabled (NER-only mode)
with nlp.select_pipes(disable=['parser']):
    start   = time.time()
    processed = list(nlp.pipe(docs_10k, batch_size=128, n_process=4))
    elapsed = time.time() - start
    print(f"NER only:      {len(docs_10k)} docs in {elapsed:.1f}s ({len(docs_10k)/elapsed:.0f} docs/s)")

# Scenario C: Tokenize only — maximum throughput
tokenizer = nlp.tokenizer
start     = time.time()
tokenized = [tokenizer(text) for text in docs_10k]
elapsed   = time.time() - start
print(f"Tokenize only: {len(docs_10k)} docs in {elapsed:.1f}s ({len(docs_10k)/elapsed:.0f} docs/s)")
```

### Memory-Efficient Streaming

```python
import spacy

nlp = spacy.load('en_core_web_sm')

def stream_entities(filepath: str, batch_size: int = 256):
    """
    Stream NER from a large file without loading all text into memory.
    Yields (line_number, entities) tuples.
    """
    def text_generator():
        with open(filepath) as f:
            for line in f:
                yield line.strip()

    for i, doc in enumerate(nlp.pipe(text_generator(), batch_size=batch_size)):
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            yield i, entities

# ─── Usage ───
# for line_num, ents in stream_entities('large_corpus.txt'):
#     for text, label in ents:
#         print(f"  Line {line_num}: {text} [{label}]")
```

## Common Pitfalls and Solutions

### Sentence Segmentation Requires Parser

```python
import spacy

nlp = spacy.load('en_core_web_sm')

# ❌ PROBLEM: Disabling parser breaks sentence segmentation
with nlp.select_pipes(disable=['parser']):
    doc = nlp("Hello world. This is sentence two.")
    sents = list(doc.sents)   # ValueError: no sentence boundaries set

# ✅ SOLUTION 1: Use sentencizer (rule-based, no parser needed)
nlp.add_pipe('sentencizer')
with nlp.select_pipes(disable=['parser']):
    doc   = nlp("Hello world. This is sentence two.")
    sents = list(doc.sents)   # Works — splits on punctuation

# ✅ SOLUTION 2: Keep parser if sentence segmentation is needed
doc   = nlp("Hello world. This is sentence two.")     # Full pipeline
sents = list(doc.sents)   # Accurate, parser-based
```

### Entity Boundaries and Spans

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Barack Obama visited the United States Capitol.")

# ❌ PROBLEM: Slicing by character offset misaligns on token boundaries
text_slice = doc.text[7:13]   # "Obama " — includes trailing space, not a valid token

# ✅ SOLUTION: Use character offsets with doc.char_span()
span = doc.char_span(7, 12, label="PERSON")   # Returns a Span object aligned to tokens
if span:
    print(span.text, span.label_)              # "Obama" PERSON

# ─── Extracting context around an entity ───
for ent in doc.ents:
    # Window of 2 tokens on each side
    start = max(0, ent.start - 2)
    end   = min(len(doc), ent.end + 2)
    window = doc[start:end]
    print(f"{ent.label_:10s} {ent.text:20s} context: \"{window.text}\"")
```

### NLTK Data Not Found

```python
# ❌ PROBLEM: LookupError: Resource punkt not found.
import nltk
from nltk.tokenize import word_tokenize
word_tokenize("Hello world")   # LookupError

# ✅ SOLUTION: Download required resources (run once per environment)
nltk.download('punkt_tab')                          # Tokenizer
nltk.download('averaged_perceptron_tagger_eng')     # POS tagger
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('brown')                              # Brown corpus
nltk.download('gutenberg')                          # Gutenberg corpus

# Or download everything (large, ~3GB):
# nltk.download('all')
```

### Multilingual Text

```python
import spacy

# ─── Load language-specific models ───
nlp_en = spacy.load('en_core_web_sm')   # English
nlp_de = spacy.load('de_core_news_sm')  # German
nlp_fr = spacy.load('fr_core_news_sm')  # French
nlp_es = spacy.load('es_core_news_sm')  # Spanish

# ─── Language detection → route to correct model ───
# (Requires langdetect or langid package)
# pip install langdetect

from langdetect import detect

MODELS = {
    'en': nlp_en,
    'de': nlp_de,
    'fr': nlp_fr,
    'es': nlp_es,
}

def process_multilingual(text: str):
    lang = detect(text)
    nlp  = MODELS.get(lang, nlp_en)   # Fallback to English
    doc  = nlp(text)
    return {
        'language': lang,
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'sentences': [sent.text for sent in doc.sents]
    }

# result = process_multilingual("Google erwärdete einen Umsatz von 300 Milliarden.")
# → {'language': 'de', 'entities': [('Google', 'ORG'), ...], ...}
```

---

spaCy and NLTK are complements: spaCy is the processing engine, NLTK is the analysis toolkit. The standard workflow is **spaCy for the pipeline** (tokenize → tag → parse → NER) **and NLTK for the linguistics** (frequency, collocations, corpus analysis, sentiment). Together they cover everything from quick entity extraction to deep corpus-level research.