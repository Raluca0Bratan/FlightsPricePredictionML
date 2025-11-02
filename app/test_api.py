import requests

# Flask API endpoint
API_URL = "http://127.0.0.1:5001/predict"

# Example JSON input (make sure it matches your model’s expected columns)
sample_data = {
    "Airline": 3,
    "AirlineID": 3,
    "Source": 0,
    "Destination": 3,
    "Total_Stops": 0,
    "DurationMinutes": 0.034111,
    "DayOfWeek": 6,
    "IsWeekend": 1,
    "DepartureHour": 0.956522,
    "DepartureDay": 24,
    "DepartureMonth": 3,
    "DepartureWeekday": 6
}

# Send POST request
response = requests.post(API_URL, json=sample_data)

# Display the result
if response.status_code == 200:
    print("✅ Prediction result:")
    print(response.json())
else:
    print("❌ Error:")
    print(response.text)
