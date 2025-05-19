import requests
import pandas as pd
import json
# Set the API endpoint
api_version = 'v1'  # Replace with the correct API version if different
base_url = 'https://api.mouser.com/api'
endpoint = f'/{api_version}/search/partnumber'

# Update with your actual part number and API key
part_number = 'LCM300L'
api_key = '653a2ac1-7c26-466d-9cf9-7ccc58195a6d'  # API key for authentication


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
 
class TLSAdapter(HTTPAdapter):
    def __init__(self, tls_version=None, **kwargs):
        self.tls_version = tls_version
        self.context = create_urllib3_context()
        if self.tls_version:
            self.context.minimum_version = self.tls_version
        super().__init__(**kwargs)
        
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = self.context
        return super().init_poolmanager(*args, **kwargs)
    
    def proxy_manager_for(self, *args, **kwargs):
        kwargs['ssl_context'] = self.context
        return super().proxy_manager_for(*args, **kwargs)
    
# Set up the adapter
session = requests.Session()
adapter = TLSAdapter()
session.mount('https://', adapter)


def getPrice(df):
    df=df.reset_index(drop=True)
    """
    Fetch price details from Mouser API for each part in the DataFrame and append the details.
    
    Args:
    df (pd.DataFrame): DataFrame containing part identifiers.
    api_url (str): Base URL for the Mouser API.
    headers (dict): Headers to include in the API requests, e.g., for authentication.

    Returns:
    pd.DataFrame: Updated DataFrame with price details appended.
    """
    for index, row in df.iterrows():
    # Headers and payload setup
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'  # Assuming Bearer token, adjust if different
        }
        
        data = {
            "SearchByPartRequest": {
                "mouserPartNumber": row['PartNumber'],  # Example part number
                "partSearchOptions": "string"  # Update or remove based on actual API requirements
            }
        }
        print(f"Requested data: {data}")
        # Perform the POST request
        response = session.post(f'{base_url}{endpoint}?apikey={api_key}', json=data, headers=headers)
        print(f"Mouser API response {response}")
        # Check the response
        try:
            if response.status_code == 200:
                # Assuming the response contains JSON data
                print("in response 200")
                data = response.json()
                print(data)
                print(index)
                
                if data['SearchResults']['Parts']:                   
                    df.at[index, 'Mouser: MouserPartNumber'] = data['SearchResults']['Parts'][0]['MouserPartNumber']
                    df.at[index, 'Mouser: Quantity'] = data['SearchResults']['Parts'][0]['PriceBreaks'][0]['Quantity']
                    df.at[index, 'Mouser: Price'] = data['SearchResults']['Parts'][0]['PriceBreaks'][0]['Price']
                    df.at[index, 'Mouser: Currency'] = data['SearchResults']['Parts'][0]['PriceBreaks'][0]['Currency']
                else :                    
                    df.at[index, 'Mouser: MouserPartNumber'] = data['SearchResults'][0]['MouserPartNumber']            
                    df.at[index, 'Mouser: Quantity'] = data['SearchResults'][0]['PriceBreaks'][0]['Quantity']
                    df.at[index, 'Mouser: Price'] = data['SearchResults'][0]['PriceBreaks'][0]['Price']
                    df.at[index, 'Mouser: Currency'] = data['SearchResults'][0]['PriceBreaks'][0]['Currency']                    
            else:
                print(f'Failed to fetch part number: {response.status_code} - {response.text}')
                df.at[index, 'Quantity'] = None
                df.at[index, 'Price'] = None
            print(df)
        except Exception as e:
            print(e)
    return df
