import json

# Load your knowledge base
# with open('symptoms.json', 'r') as f:
#     symptoms_db = json.load(f)
    
# with open('/nutrient.json', 'r') as f:
#     nutrients_db = json.load(f)
    
# with open('solutions.json', 'r') as f:
#     solutions_db = json.load(f)

def get_nutrient_recommendations(identified_symptoms):
    """Generate nutrient recommendations based on identified symptoms"""
    with open('symptoms.json', 'r') as f:
        symptoms_db = json.load(f)
        
    with open('nutrient.json', 'r') as f:
        nutrients_db = json.load(f)
        
    with open('solutions.json', 'r') as f:
        solutions_db = json.load(f)
    
    # Track potential recommendations and their relevance scores
    potential_recommendations = {}
    
    # For each identified symptom
    for symptom in identified_symptoms:
        symptom_id = symptom
        
        # Find matching solutions in solutions_db
        matching_solutions = [s for s in solutions_db["relationships"] if s["symptomId"] == symptom_id]
        
        # For each potential solution for this symptom
        for solution in matching_solutions:
            nutrient_id = solution["nutrientId"]
            
            # Calculate a relevance score for this recommendation
            # Based on symptom confidence, evidence strength, and context match
            evidence_strength = {
                "High": 3,
                "Moderate": 2,
                "Low": 1
            }.get(solution["evidenceStrength"], 1)
            
            # Adjust for context if needed (severity, duration)
            # context_modifier = 1.0
            # if solution.get("conditions"):
            #     # Example: Solution works better for chronic cases
            #     if "chronic" in solution["conditions"].lower() and context["duration"] == "chronic":
            #         context_modifier = 1.5
            #     # Example: Solution works better for severe cases    
            #     if "severe" in solution["conditions"].lower() and context["severity"] == "severe":
            #         context_modifier = 1.5
            
            # Calculate final score
            relevance_score = evidence_strength
            
            # Add to potential recommendations or update if already exists
            if nutrient_id in potential_recommendations:
                # Take the higher score if nutrient already recommended
                potential_recommendations[nutrient_id] = max(
                    potential_recommendations[nutrient_id], 
                    relevance_score
                )
            else:
                potential_recommendations[nutrient_id] = relevance_score
    
    # Get nutrient details for the top recommendations
    sorted_nutrients = sorted(
        potential_recommendations.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Prepare final recommendations (top 3-5)
    final_recommendations = []
    for nutrient_id, score in sorted_nutrients[:5]:  # Adjust number as needed
        # Find nutrient details
        nutrient_details = next((n for n in nutrients_db["nutrients"] if n["id"] == nutrient_id), None)
        
        if nutrient_details:
            # Get the matching solutions for explanation
            related_symptoms = []
            for symptom in identified_symptoms:
                matching_solutions = [s for s in solutions_db["relationships"] 
                                    if s["symptomId"] == symptom 
                                    and s["nutrientId"] == nutrient_id]
                
                if matching_solutions:
                    symptom_name = next((s["Name"] for s in symptoms_db["symptoms"] if s["ID"] == symptom), "unknown")
                    related_symptoms.append({
                        "symptom": symptom_name,
                        "mechanism": matching_solutions[0].get("mechanism", "")
                    })
            
            final_recommendations.append({
                "nutrient_id": nutrient_id,
                "name": nutrient_details["name"],
                "category": nutrient_details["category"],
                "dosage": nutrient_details["dosage"]["standard"],
                "relevance_score": score,
                "related_symptoms": related_symptoms,
                "contraindications": nutrient_details.get("contraindications", []),
                "interactions": nutrient_details.get("potential_interactions", [])
            })
    
    return final_recommendations