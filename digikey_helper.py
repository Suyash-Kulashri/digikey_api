import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def authDigi():    
    # Your client credentials
    client_id = os.getenv("DIGIKEY_CLIENT_ID")
    print(f"client_id: {client_id}")
    client_secret = os.getenv("DIGIKEY_CLIENT_SECRET") 

    # DigiKey OAuth 2.0 Token Endpoint
    token_url = os.getenv("DIGIKEY_TOKEN_URL") 

    # Request payload
    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials'
    }

    # Make the POST request to get the access token
    print("Sending request to DigiKey for auth token...")
    response = requests.post(token_url, data=payload, verify=False)

    # Debugging: Check the response status
    print(f"Response status code: {response.status_code}")
   
    # Parse the JSON response
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get('access_token', None)
        # Debugging: Print the access token
        print(f"Access token: {access_token}")
        return access_token
    else:
        # Print the error response for debugging
        print(f"Error: {response.status_code}, {response.text}")
        return "Error"
    
def getPrice(similarItems):
    # Authenticate and get the token
    token = authDigi()
    print(f"Token returned: {token}")

    # DigiKey API endpoint template
    url_template = 'https://api.digikey.com/products/v4/search/{}/pricing'

    # Prepare new columns in the DataFrame for the results
    similarItems['Digikey: BreakQuantity'] = None
    similarItems['Digikey: UnitPrice'] = None
    similarItems['Digikey: TotalPrice'] = None

    # Headers for the API request
    headers = {
        'Authorization': f'Bearer {token}',
        'X-DIGIKEY-Client-Id': os.getenv("DIGIKEY_CLIENT_ID"),
        'Content-Type': 'application/json'
    }

    # Iterate through each row in the DataFrame to get pricing for each "Model"
    for idx, row in similarItems.iterrows():
        model = row['PartNumber']
        url = url_template.format(model)
        
        # Send the GET request to DigiKey's API
        response = requests.get(url, headers=headers)

        print(f"Request for model {model}: {response.status_code}")
        response_data = response.json()
        print(response_data)
        if response.status_code == 200:
            response_data = response.json()
            print(response_data)
            # Check if ProductPricing exists in the response
            if 'ProductPricings' in response_data and len(response_data['ProductPricings']) > 0:
                pricing_info = response_data['ProductPricings'][0]
                
                # Extract pricing details
                if 'ProductVariations' in pricing_info and len(pricing_info['ProductVariations']) > 0:
                    pricing_variation = pricing_info['ProductVariations'][0]
                    if 'StandardPricing' in pricing_variation and len(pricing_variation['StandardPricing']) > 0:
                        price_details = pricing_variation['StandardPricing'][0]
                        
                        # Add pricing details to the DataFrame
                        similarItems.at[idx, 'Digikey: BreakQuantity'] = price_details.get('BreakQuantity')
                        similarItems.at[idx, 'Digikey: UnitPrice'] = price_details.get('UnitPrice')
                        similarItems.at[idx, 'Digikey: TotalPrice'] = price_details.get('TotalPrice')
        else:
            # If there was an error, you can handle it (log, retry, or skip)
            print(f"Error fetching data for {model}: {response.status_code}")

    # Return the updated DataFrame
    return similarItems

    # print (similarItems['Model'])

# df= pd.read_csv("data/testDigikey.csv")

# print(getPrice(df))
    