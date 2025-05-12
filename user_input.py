from recommend_nutrient import *
import joblib
from preprocess_text import *

def process_user_input(user_text):
    """Process user input and return nutrient recommendations"""

    improved_pipeline = joblib.load('symptom_model.pkl')
    mlb = joblib.load("mlb.pkl")

    user_text = improved_preprocess_text(user_text)
    prediction = improved_pipeline.predict([user_text])
    predicted_symptoms = mlb.inverse_transform(prediction)[0]
    print(predicted_symptoms)

    
    # Step 1: Use your trained ML model to identify symptoms
    
    # Step 2: Generate recommendations based on identified symptoms
    recommendations = get_nutrient_recommendations(
        predicted_symptoms
    )
    
    # Step 3: Generate explanations for each recommendation
    # for recommendation in recommendations:
    #     recommendation["explanation"] = generate_explanation(recommendation)
    
    # Prepare final response
    with open('symptoms.json', 'r') as f:
        symptoms_db = json.load(f)
    response = {
        "original_text": user_text,
        "identified_symptoms": [
            {"id": s} 
            for s in predicted_symptoms
        ],
        "recommendations": recommendations
    }
    print(recommendations)
    
    return response

user_text = input("What are you struggling with: ")
process_user_input(user_text)