import requests
import os
import pandas as pd
from dotenv import load_dotenv
import certifi

load_dotenv()

def authDigi():    
    client_id = os.getenv("DIGIKEY_CLIENT_ID")
    client_secret = os.getenv("DIGIKEY_CLIENT_SECRET") 
    token_url = os.getenv("DIGIKEY_TOKEN_URL") 

    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials',
        'scope': 'read_product_information'
    }

    response = requests.post(token_url, data=payload, verify=certifi.where())
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get('access_token', None)
        return access_token
    else:
        return "Error"
    
def getPrice(similarItems):
    token = authDigi()
    if token == "Error":
        return similarItems

    url_template = 'https://api.digikey.com/products/v4/search/{productNumber}/pricing'

    similarItems['Digikey: BreakQuantity'] = None
    similarItems['Digikey: UnitPrice'] = None
    similarItems['Digikey: TotalPrice'] = None

    headers = {
        'Authorization': f'Bearer {token}',
        'X-DIGIKEY-Client-Id': os.getenv("DIGIKEY_CLIENT_ID")
    }

    for idx, row in similarItems.iterrows():
        model = row['PartNumber']
        url = url_template.format(productNumber=model)
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            if 'ProductPricings' in response_data and len(response_data['ProductPricings']) > 0:
                product_pricing = response_data['ProductPricings'][0]
                if 'ProductVariations' in product_pricing and len(product_pricing['ProductVariations']) > 0:
                    variation = product_pricing['ProductVariations'][0]
                    if 'StandardPricing' in variation and len(variation['StandardPricing']) > 0:
                        price_details = variation['StandardPricing'][0]
                        similarItems.at[idx, 'Digikey: BreakQuantity'] = price_details.get('BreakQuantity')
                        similarItems.at[idx, 'Digikey: UnitPrice'] = price_details.get('UnitPrice')
                        similarItems.at[idx, 'Digikey: TotalPrice'] = price_details.get('TotalPrice')

    return similarItems

if __name__ == "__main__":
    data = {
        'PartNumber': [
            'LRS-35-5', 'LRS-35-12', 'LRS-35-15', 'LRS-35-24', 'LRS-35-36', 'LRS-35-48',
            'LRS-50-3.3', 'LRS-50-5', 'LRS-50-12', 'LRS-50-15', 'LRS-50-24', 'LRS-50-36', 'LRS-50-48',
            'LRS-75-5', 'LRS-75-12', 'LRS-75-15', 'LRS-75-24', 'LRS-75-36', 'LRS-75-48'
        ]
    }
    df = pd.DataFrame(data)
    result = getPrice(df)
    print(result.to_string(index=False))