from shared_imports import *
from preprocess_text import *

improved_pipeline = joblib.load('./symptom_model.pkl')
mlb = joblib.load("./mlb.pkl")

test_text = "I am struggling to focus in class"
processed_test = improved_preprocess_text(test_text)
prediction = improved_pipeline.predict([processed_test])
predicted_symptoms = mlb.inverse_transform(prediction)[0]

print(f"Text: {test_text}")
print(prediction)
print(f"Predicted symptoms: {predicted_symptoms}")