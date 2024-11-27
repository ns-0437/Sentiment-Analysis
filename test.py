import requests

# URL of your FastAPI server
url = 'http://127.0.0.1:8080/predict'

# Example review to test
review = "This product exceeded my expectations!"

# Send POST request with JSON body
try:
    response = requests.get(url, json={"review": review})

    # Check if the request was successful
    response.raise_for_status()

    # Print the status code
    print("Response Status Code:", response.status_code)

    # Try to parse JSON response
    try:
        json_response = response.json()
        print("Response JSON:", json_response)
    except ValueError as e:
        print("Response is not in JSON format:", e)

except requests.exceptions.HTTPError as err:
    print("HTTP Error:", err)

except requests.exceptions.RequestException as e:
    print("Error during request:", e)