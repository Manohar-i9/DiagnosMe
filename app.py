from flask import Flask,render_template, request, jsonify
from flask_cors import CORS, cross_origin
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

symptom_vectors = np.load('d:\\UWindsor\\AAI\\Project\\DiagnosMe Project\\symptom_vectors2.npy')

model = SentenceTransformer('all-mpnet-base-v2')

df_diseases_symptoms = pd.read_csv('d:\\UWindsor\\AAI\\Project\\DiagnosMe Project\\Dataset/disease-symptoms.csv')
df_diseases_precautions = pd.read_csv('d:\\UWindsor\\AAI\\Project\\DiagnosMe Project\\Dataset/disease-precautions.csv')
df_diseases_medications = pd.read_csv('d:\\UWindsor\\AAI\\Project\\DiagnosMe Project\\Dataset/disease-medications.csv')

df_diseases_symptoms = df_diseases_symptoms.replace(np.nan, '')
symptoms = df_diseases_symptoms.drop(columns=['Disease'])
symptoms['sym'] = ''
for i in range(1, 18):
  symptoms['sym'] += symptoms['Symptom_' + str(i)]

symptoms['sym'] = symptoms['sym'].replace('_', ' ', regex=True)
df_diseases_symptoms = pd.concat([df_diseases_symptoms['Disease'], symptoms['sym']], axis=1)
df_diseases_symptoms = df_diseases_symptoms.rename(columns={'Disease': 'disease', 'sym': 'symptoms'})

df_diseases_precautions = df_diseases_precautions.replace(np.nan, '')
precautions = df_diseases_precautions.drop(columns=['Disease'])
precautions['sym'] = ''
for i in range(1, 5):
  precautions['sym'] += precautions['Precaution_' + str(i)] + ','

df_diseases_precautions = pd.concat([df_diseases_precautions['Disease'], precautions['sym']], axis=1)
df_diseases_precautions = df_diseases_precautions.rename(columns={'Disease': 'disease', 'sym': 'precautions'})

def cosine_similarity(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def sort(tup):
  tup.sort(key = lambda x: x[1])
  return tup

# Securely load your Google API key from an environment variable or other secure sources
GOOGLE_API_KEY = 'AIzaSyAy01q6C6bjES_vLwwtBU-sFpzYvWDMLkE'

def find_hospitals(city_name):
    clean_city_name = city_name.lower().strip()  # Normalize city name input for better matching

    # Step 1: Geocode the city name to get latitude and longitude
    geocode_url = f'https://maps.googleapis.com/maps/api/geocode/json?address={clean_city_name}&key={GOOGLE_API_KEY}'
    geocode_response = requests.get(geocode_url)
    geocode_data = geocode_response.json()

    if geocode_data['status'] == 'OK':
        lat = geocode_data['results'][0]['geometry']['location']['lat']
        lon = geocode_data['results'][0]['geometry']['location']['lng']
        
        # Step 2: Use the lat and lon to search for hospitals
        places_url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=4000&type=hospital&key={GOOGLE_API_KEY}'
        places_response = requests.get(places_url)
        places_data = places_response.json()
        
        if places_data['status'] == 'OK':
            # Limit the number of hospitals to 10
            hospitals = [{
                'name': place['name'],
                'address': place.get('vicinity')
            } for place in places_data['results']][:10]  # Adjust the slice here to limit to 10
            return hospitals, ""
        else:
            print(f"Places API error: {places_data.get('error_message', 'No error message')}")
    else:
        print(f"Geocoding API error: {geocode_data.get('error_message', 'No error message')}")
    return [], "Couldn't find any hospitals for this location."
  
@app.route('/getResults', methods=['GET'])
@cross_origin()
def get_results():
    query_params = request.args
    input = query_params.to_dict()['query']
    city = query_params.to_dict()['city']
    input_vector = model.encode(input)
    cos_sim = []

    for i in range(symptom_vectors.shape[0]):
        cos_sim.append((df_diseases_symptoms['disease'][i], cosine_similarity(input_vector, symptom_vectors[i])))

    sorted_cos_sim = sort(cos_sim)

    diseases = []
    n = len(sorted_cos_sim)
    for j in range(n):
        i = n - j - 1
        is_present = False
        for d in diseases:
            if d == sorted_cos_sim[i][0]:
                is_present = True
                break

        if not is_present:
            diseases.append(sorted_cos_sim[i][0])
            
    precautions = df_diseases_precautions[df_diseases_precautions['disease'] == sorted_cos_sim[-1][0]]['precautions'].to_list()[0]
    medications = df_diseases_medications[df_diseases_medications['Disease'] == sorted_cos_sim[-1][0].strip()]['Common_Medications'].to_list()

    hospitals, _ = find_hospitals(city)
    response = {
        "query": input,
        "diseases": diseases[:5],
        "precautions": precautions,
        "medications": medications,
        "most_probable_disease": sorted_cos_sim[-1][0].strip(),
        "hospitals": hospitals
    }
    
    return jsonify(response), 200
  
def home():
    hospitals = []
    message = ""
    city_name = ""  # Initialize city_name variable
    if request.method == 'POST':
        city_name = request.form['city_name']
        hospitals, message = find_hospitals(city_name)
        if not hospitals:  # If the list is empty, set a message
            message = "Couldn't find any hospitals for this location."
    # Pass the city_name to the template
    return render_template('index.html', hospitals=hospitals, message=message, city_name=city_name)

if __name__ == '__main__':
    app.run(debug=True)
