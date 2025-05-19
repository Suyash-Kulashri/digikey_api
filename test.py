import streamlit as st
import requests
import os

# Set page title
st.title("DigiKey Product Price Checker")

# Input field for product name
product_name = st.text_input("Enter Product Name", placeholder="e.g., Arduino Uno")

# Button to trigger API call
if st.button("Search Price"):
    if not product_name:
        st.error("Please enter a product name.")
    else:
        try:
            # Retrieve API key from Streamlit secrets or environment variable
            api_key = st.secrets.get("DIGIKEY_API_KEY") or os.getenv("DIGIKEY_API_KEY")
            
            if not api_key:
                st.error("API key not configured. Please set DIGIKEY_API_KEY in secrets.")
                st.stop()

            # DigiKey API endpoint for product search
            url = "https://api.digikey.com/services/partsearch/v3/keywordsearch"
            
            # Set headers with API key
            headers = {
                "Authorization": f"Bearer {api_key}",
                "X-DIGIKEY-Client-Id": st.secrets.get("DIGIKEY_CLIENT_ID") or os.getenv("DIGIKEY_CLIENT_ID"),
                "Content-Type": "application/json"
            }
            
            # Payload for keyword search
            payload = {
                "keywords": product_name,
                "limit": 1,  # Limit to one result for simplicity
                "offset": 0
            }
            
            # Make API request
            response = requests.post(url, json=payload, headers=headers)
            
            # Check for successful response
            if response.status_code == 200:
                data = response.json()
                products = data.get("products", [])
                
                if products:
                    product = products[0]
                    name = product.get("productDescription", "N/A")
                    price = product.get("standardPricing", [{}])[0].get("unitPrice", "N/A")
                    
                    # Display results
                    st.success("Product found!")
                    st.write(f"**Product Name:** {name}")
                    st.write(f"**Price:** ${price:.2f}" if isinstance(price, (int, float)) else f"**Price:** {price}")
                else:
                    st.warning("No products found for the given name.")
            else:
                st.error(f"API request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Instructions for setup
st.markdown("""
### Setup Instructions
1. **Obtain DigiKey API Credentials**:
   - Register for a DigiKey API account at [DigiKey API Portal](https://developer.digikey.com/).
   - Get your `Client ID` and `API Key`.
2. **Configure Secrets**:
   - In your Streamlit app, create a `secrets.toml` file or configure secrets in Streamlit Cloud with:
     ```toml
     DIGIKEY_API_KEY = "your_api_key_here"
     DIGIKEY_CLIENT_ID = "your_client_id_here"
     ```
   - Alternatively, set environment variables `DIGIKEY_API_KEY` and `DIGIKEY_CLIENT_ID`.
3. **Install Dependencies**:
   - Run `pip install streamlit requests`.
4. **Run the App**:
   - Save this code as `digikey_price_checker.py` and run `streamlit run digikey_price_checker.py`.
""")