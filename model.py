from shared_imports import *

class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return np.array([self.model.encode(text) for text in X])

advanced_pipeline = Pipeline([
    ('embeddings', SentenceTransformerEncoder()),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
])


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Replace contractions
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"i've", "i have", text)
    text = re.sub(r"it's", "it is", text)
    # Add more contractions as needed
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords (optional - might lose important context for symptom descriptions)
    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

    return text

def synonym_replacement(text, n=1):
    words = text.split()
    if len(words) <= 3:  # Don't augment very short texts
        return text
        
    new_words = words.copy()
    random_word_indices = random.sample(range(len(words)), min(n, len(words)))
    
    # Simple synonym mapping - expand this with domain knowledge
    synonyms = {
        'tired': ['exhausted', 'fatigued', 'weary', 'drained'],
        'pain': ['ache', 'soreness', 'discomfort', 'hurt'],
        'headache': ['migraine', 'head pain', 'throbbing head', 'skull ache'],
        'bloated': ['swollen', 'puffy', 'distended', 'gassy'],
        'anxious': ['worried', 'nervous', 'uptight', 'on edge'],
        'sad': ['depressed', 'down', 'blue', 'unhappy'],
        # Add more symptom-specific synonyms
    }
    
    for idx in random_word_indices:
        word = words[idx]
        if word in synonyms:
            new_words[idx] = random.choice(synonyms[word])
    
    return ' '.join(new_words)

def augment_with_nlpaug(text):
    # Contextual word embeddings replacement
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', 
        action="substitute",
        aug_p=0.3
    )
    augmented_text = aug.augment(text)
    return augmented_text

def random_swap(text, n=1):
    words = text.split()
    if len(words) <= 3:
        return text
        
    new_words = words.copy()
    for _ in range(min(n, len(words)-1)):
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    
    return ' '.join(new_words)

# Upsample minority classes or downsample majority classes
def balance_dataset(data, target_counts=None):
    # Group examples by symptom
    symptom_groups = {}
    for symptom in symptom_counts.keys():
        symptom_groups[symptom] = [ex for ex in data if symptom in ex['labels']['symptoms']]
    
    # Determine target count (if not provided)
    if target_counts is None:
        # Use median count as target
        median_count = np.median(list(symptom_counts.values()))
        target_counts = {symptom: int(median_count) for symptom in symptom_counts.keys()}
    
    balanced_data = []
    for symptom, target in target_counts.items():
        group = symptom_groups[symptom]
        current_count = len(group)
        
        if current_count < target:
            # Upsample
            upsampled = resample(
                group,
                replace=True,
                n_samples=target,
                random_state=42
            )
            balanced_data.extend(upsampled)
        elif current_count > target:
            # Downsample
            downsampled = resample(
                group,
                replace=False,
                n_samples=target,
                random_state=42
            )
            balanced_data.extend(downsampled)
        else:
            balanced_data.extend(group)
    
    # Remove duplicates
    unique_texts = set()
    unique_data = []
    for example in balanced_data:
        if example['user_text'] not in unique_texts:
            unique_texts.add(example['user_text'])
            unique_data.append(example)
    
    return unique_data

def improved_preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Replace contractions
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"i've", "i have", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"feeling", "I feel", text)
    text = re.sub(r"foggy", "brain fog", text)
    text = re.sub(r"groggy", "tired", text)
    text = re.sub(r"get", "feel", text)
    text = re.sub(r"am struggling to", "cannot", text)
    text = re.sub(r"i feel", "", text)
    # Add more contractions as needed
    
    # Remove filler phrases that might distract the model
    text = re.sub(r"i (am|have been|feel|keep|constantly|always) ", " ", text)
    text = re.sub(r"(experiencing|having|suffering from|dealing with) ", " ", text)
    
    # Highlight symptom indicators by repeating them
    symptom_indicators = [
        "tired", "exhausted", "fatigue", "foggy", "cloudy", "memory", "forget", 
        "sleep", "insomnia", "headache", "pain", "ache", "sore", "anxiety", 
        "stress", "mood", "depressed", "inflamed", "swelling", "bloated", 
        "constipation", "diarrhea", "reflux", "heartburn", "dry", "thirsty", 
        "dizzy", "cravings", "weight", "hormone", "hair", "skin", "acne", 
        "itchy", "muscle", "joint", "concentration", "focus", "energy", "sleep",
        "sad", "lost", "groggy"
    ]
    
    for indicator in symptom_indicators:
        if indicator in text:
            # Repeat important symptom words to give them more weight
            text = text.replace(indicator, f"{indicator} {indicator} {indicator}")
    
    # Remove punctuation and numbers (optional)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text


with open('./user_more.json', 'r') as f:
  training_data = json.load(f)
  print("file opened")

all_symptom_ids = [
    "SLE02", "SLE03", "SLE04", "FEM01", "FEM02", "FEM03", "IMM03", "IMM04", 
    "HAI02", "HAI03", "COG01", "COG02", "DIG04", "DIG05", "SKN03", "SKN04", 
    "HYD02", "HYD03", "FAT01", "FOG01", "SLE01", "DIG01", "DIG02", "DIG03", 
    "STR01", "MOO01", "ANX01", "IMM01", "IMM02", "HOR01", "HOR02", "PAI01", 
    "PAI02", "PAI03", "SKN01", "SKN02", "HAI01", "WGT01", "CRV01", "ENE01", 
    "ENE02", "IMB01", "BLD01", "VIT01", "MET01", "MAS01", "MEN01", "HYD01", 
    "STR02", "EXE01"
]

symptom_counts = Counter()
for example in training_data:
    for sym in example['labels']['symptoms']:
      symptom_counts[sym] += 1

print(symptom_counts)

processed_data = []
for example in training_data:
    processed_text = preprocess_text(example['user_text'])
    processed_data.append({
        'user_text': processed_text,
        'labels': example['labels']
    })

augmented_data = []
for example in processed_data:
    text = example['user_text']
    labels = example['labels']
    
    # Create a few variations with different augmentation techniques
    augmented_data.append({
        'user_text': synonym_replacement(text),
        'labels': labels
    })
    
    augmented_data.append({
        'user_text': random_swap(text),
        'labels': labels
    })

all_data = processed_data + augmented_data

balanced_data = balance_dataset(all_data)

X = [example['user_text'] for example in balanced_data]
y = [example['labels']['symptoms'] for example in balanced_data]

mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train_improved = [improved_preprocess_text(text) for text in X_train]
X_test_improved = [improved_preprocess_text(text) for text in X_test]

improved_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        analyzer='word',
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        binary=True,         # Binary features work well
        min_df=2,            # Minimum document frequency
        max_df=0.95          # Maximum document frequency
    )),
    ('classifier', MultiOutputClassifier(LogisticRegression(
        C=3,                 # Adjust regularization strength
        class_weight='balanced',  # Handle class imbalance
        max_iter=2000,       # More iterations for convergence
        solver='liblinear',  # Works well with smaller datasets
        dual=False           # Better for cases where n_samples > n_features
    )))
])

improved_pipeline.fit(X_train_improved, y_train)
y_pred = improved_pipeline.predict(X_test_improved)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

joblib.dump(improved_pipeline, 'symptom_model.pkl')
joblib.dump(mlb, 'mlb.pkl')

newimproved_pipeline = joblib.load('./symptom_model.pkl')
test_text = "I am struggling to focus in class"
processed_test = improved_preprocess_text(test_text)
prediction = newimproved_pipeline.predict([processed_test])
predicted_symptoms = mlb.inverse_transform(prediction)[0]

print(f"Text: {test_text}")
print(prediction)
print(f"Predicted symptoms: {predicted_symptoms}")