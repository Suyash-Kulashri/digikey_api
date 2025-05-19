import streamlit as st
import numpy as np
import pandas as pd
from app.faiss_index import get_faiss_index
from app.database import get_aei_parts, get_data_for_faiss, get_numerical_values, get_manufacturers
import app.digikey_helper as digikey
import app.mouser_helper as mouser
import matplotlib.pyplot as plt
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.markdown("""
    <style>
        /* Professional scrollbar styling */
        div[data-testid="stDataFrame"] div {
            scrollbar-width: thin !important;
            scrollbar-color: #555555 #f0f2f6 !important;
        }
        
        div[data-testid="stDataFrame"] div::-webkit-scrollbar {
            height: 8px !important;
            width: 8px !important;
        }
        
        div[data-testid="stDataFrame"] div::-webkit-scrollbar-track {
            background: #f0f2f6 !important;
            border-radius: 4px !important;
        }
        
        div[data-testid="stDataFrame"] div::-webkit-scrollbar-thumb {
            background-color: #555555 !important;
            border-radius: 4px !important;
            border: 2px solid #f0f2f6 !important;
        }
        
        /* Only show scrollbar when needed */
        div[data-testid="stDataFrame"] > div[style*="overflow-x: auto"] {
            overflow-x: auto !important;
            padding-bottom: 4px !important;
        }
        
        /* Fix for single-row tables */
        div[data-testid="stDataFrame"] > div[style*="overflow-x: auto"]:not(:hover) {
            overflow-x: hidden !important;
        }
        
        div[data-testid="stDataFrame"] > div[style*="overflow-x: auto"]:hover {
            overflow-x: auto !important;
        }
        
        /* Frozen headers */
        .stDataFrame thead th {
            position: sticky !important;
            top: 0 !important;
            background-color: white !important;
            z-index: 100 !important;
            border-bottom: 2px solid #e0e0e0 !important;
        }
        
        /* Row dividers */
        .stDataFrame tbody tr td {
            border-bottom: 8px solid #f0f0f0 !important;
        }
    </style>
""", unsafe_allow_html=True)

def format_column_name(column_name):
    """Convert camelCase or PascalCase column names to spaced display names."""
    special_cases = {
        'PartNumber': 'Part Number',
        'InputVoltage': 'Input Voltage',
        'OutputVoltage': 'Output Voltage',
        'OutputCurrent': 'Output Current',
        'OutputPower': 'Output Power',
        'QueryVectorRank': 'Query Vector Rank',
        'PartStatus': 'Part Status'
    }
    
    if column_name in special_cases:
        return special_cases[column_name]
    
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', column_name)

def display_table(df, title=None):
    """Enhanced table display with professional scrollbars and post-table dividers"""
    if title:
        st.markdown(f"**{title}**")
    
    display_columns = {col: format_column_name(col) for col in df.columns}
    displayed_df = df.rename(columns=display_columns)
    
    row_height = 35
    min_height = row_height * 2 + 30
    max_height = 600
    
    table_height = min_height if len(df) <= 1 else min(max_height, row_height * (len(df) + 1))
    
    st.dataframe(
        displayed_df,
        use_container_width=True,
        height=table_height
    )
    
    st.markdown("---")
    st.caption(f"Showing {len(df)} records")

@st.cache_data
def load_static_data():
    """Load static data once and cache it"""
    df, index = get_faiss_index()
    df_aei = get_aei_parts()
    manufacturers_df = get_manufacturers()
    return df, index, df_aei, manufacturers_df

DF, INDEX, DF_AEI, MANUFACTURERS_DF = load_static_data()

def calculate_radius(query_vector, part_vector):
    """Calculate the Euclidean distance (Radius) between the query vector and the part vector."""
    return np.linalg.norm(query_vector - part_vector)

def generate_conversational_summary(results_df: pd.DataFrame, original_part: str) -> str:
    """Generate a conversational summary of the top 3 cross-reference results using OpenAI."""
    if results_df.empty:
        return f"No cross-reference parts found for {original_part}."
    
    top_results = results_df.sort_values('distance').head(3)
    
    parts_info = []
    for _, row in top_results.iterrows():
        parts_info.append({
            'Manufacturer': row['Manufacturer'],
            'PartNumber': row['PartNumber'],
            'OutputVoltage': row['OutputVoltage'],
            'OutputCurrent': row['OutputCurrent'],
            'OutputPower': row['OutputPower'],
            'Distance': row['distance']
        })
    
    prompt = f"""
    You're an expert in electronic components helping an engineer find cross-reference parts.
    The original part is {original_part}. Here are the top 3 potential replacements:
    
    {json.dumps(parts_info, indent=2)}
    
    Please provide a concise, conversational summary that:
    1. Lists the top 3 replacement parts by manufacturer and part number
    2. Explains why each might be a good replacement (focus on voltage, current, power matching)
    3. Mentions any notable differences to be aware of
    4. Keeps the tone professional but friendly
    5. Uses bullet points for each part recommendation
    
    Format the response in clear paragraphs with proper technical terminology.
    """
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're an expert electronic components engineer helping with cross-references."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Couldn't generate a conversational summary for {original_part} at this time."

def plot_radar_chart(v, c, p, selected_row):
    """Plot a radar chart for selected part comparison."""
    attributes = ['OutputVoltage', 'OutputCurrent', 'OutputPower']
    num_vars = len(attributes)

    baseline_values = [v, c, p]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, baseline_values + baseline_values[:1], label='Baseline')
    ax.fill(angles, baseline_values + baseline_values[:1], alpha=0.1)
    
    selected_values = [
        selected_row['OutputVoltage'],
        selected_row['OutputCurrent'],
        selected_row['OutputPower']
    ]
    ax.plot(angles, selected_values + selected_values[:1], label='Selected Part')
    ax.fill(angles, selected_values + selected_values[:1], alpha=0.3)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([format_column_name(attr) for attr in attributes])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Radar Chart for Selected Part")
    st.pyplot(fig)

def main():
    """Main function for the Streamlit UI."""
    st.title("Cross-Reference Assistant")
    
    ranked_specs = []

    model_list = DF_AEI["PartNumber"].unique().tolist()
    manufacturers = MANUFACTURERS_DF["Manufacturer"].unique().tolist()
    
    model_list.insert(0, "Select a Model")
    
    selected_model = st.selectbox("Choose a AEI Part Number", model_list)
    
    input_voltage = st.text_input("Specify Input Voltage (Optional)", "")
    selected_number_records = st.select_slider(
        "Set the Number of Records to Display for Pricing",
        options=list(range(1, 20)))
    Manufactureroptions = st.multiselect(
        "Filter Results by Manufacturer (Optional)",
        manufacturers)
    
    if selected_model != "Select a Model":
        with st.container(border=True):
            st.subheader("Select Specifications")
            
            available_specs = [
                "OutputVoltage", "OutputCurrent", "OutputPower", "InputVoltage"
            ]
            
            selected_specs = available_specs.copy()
            
            for spec in available_specs:
                if not st.checkbox(f"Select {format_column_name(spec)}", key=spec, value=True):
                    selected_specs.remove(spec)
        
        if selected_specs:
            with st.container(border=True):
                st.subheader("Rank Specifications")
                
                ranked_specs = []
                ranking_values = {}
                
                if len(selected_specs) > 1:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.write("**Enter Rankings**")
                        for i, spec in enumerate(selected_specs):
                            rank = st.text_input(
                                f"Rank for {format_column_name(spec)} (1 to {len(selected_specs)})",
                                value=str(i + 1),
                                key=f"rank_{spec}"
                            )
                            ranked_specs.append((spec, int(rank)))
                    
                    with col2:
                        st.write("**Enter Values for Rankings**")
                        selected_model_data = DF_AEI[DF_AEI["PartNumber"] == selected_model].iloc[0]
                        
                        for i, spec in enumerate(ranked_specs):
                            default_value = selected_model_data.get(spec[0], "")
                            ranking_values[spec[0]] = st.text_input(
                                f"Enter value for {format_column_name(spec[0])} (Rank {spec[1]})",
                                value=str(default_value),
                                key=f"value_{spec[0]}"
                            )
                
                elif len(selected_specs) == 1:
                    ranked_specs = [(selected_specs[0], 1)]
                    st.write("**Enter Values for Rankings**")
                    
                    selected_model_data = DF_AEI[DF_AEI["PartNumber"] == selected_model].iloc[0]
                    
                    default_value = selected_model_data.get(ranked_specs[0][0], "")
                    ranking_values[ranked_specs[0][0]] = st.text_input(
                        f"Enter value for {format_column_name(ranked_specs[0][0])} (Rank 1)",
                        value=str(default_value),
                        key=f"value_{ranked_specs[0][0]}"
                    )
                
                if st.button("Search Ranking"):
                    try:
                        product_df = DF_AEI[
                            DF_AEI["PartNumber"].str.contains(selected_model, na=False, regex=False)
                        ]
                        
                        if "PartStatus" in DF_AEI.columns:
                            product_df = product_df[product_df["PartStatus"] == "Active"]
                        
                        if not product_df.empty:
                            row_options = [
                                f"{format_column_name('PartNumber')}: {row['PartNumber']}, "
                                f"{format_column_name('OutputVoltage')}: {row['OutputVoltage']}, "
                                f"{format_column_name('OutputCurrent')}: {row['OutputCurrent']}, "
                                f"{format_column_name('OutputPower')}: {row['OutputPower']}"
                                for _, row in product_df.iterrows()
                            ]

                            selected_option = st.radio("Choose a Result for Cross-Reference Details", row_options)
                            selected_index = row_options.index(selected_option)
                            selected_row = product_df.iloc[selected_index]

                            query_vector = np.array([
                                float(ranking_values.get("InputVoltage", selected_row['InputVoltage'])),
                                float(ranking_values.get("OutputVoltage", selected_row['OutputVoltage'])),
                                float(ranking_values.get("OutputPower", selected_row['OutputPower'])),
                                float(ranking_values.get("OutputCurrent", selected_row['OutputCurrent']))
                            ], dtype='float32').reshape(1, -1)
                            
                            st.write("Updated Query Vector:")
                            st.write(query_vector)
                            
                            k = 400
                            distances, indices = INDEX.search(query_vector, k)

                            similar_items = DF.iloc[indices[0]].copy()
                            original_indices = indices[0]

                            if "PartStatus" in DF.columns:
                                mask = similar_items["PartStatus"] == "Active"
                                similar_items = similar_items[mask]
                                distances = distances[0][mask]
                                original_indices = original_indices[mask]
                            else:
                                distances = distances[0]

                            similar_items['Radius'] = [
                                calculate_radius(query_vector, np.array([
                                    row['InputVoltage'],
                                    row['OutputVoltage'],
                                    row['OutputPower'],
                                    row['OutputCurrent']
                                ], dtype='float32'))
                                if pd.notnull(row['InputVoltage']) else None
                                for _, row in similar_items.iterrows()
                            ]
                            
                            similar_items['QueryVectorRank'] = distances
                            
                            similar_items = similar_items.sort_values(by='QueryVectorRank')

                            if Manufactureroptions:
                                similar_items = similar_items[similar_items["Manufacturer"].isin(Manufactureroptions)]

                            if ranked_specs:
                                similar_items = similar_items.sort_values(by=[spec[0] for spec in ranked_specs])

                            st.write(f"Filtered Results with Radius and Query Vector Rank ({len(similar_items)} rows):")
                            display_table(
                                similar_items[['PartNumber', 'Manufacturer', 'PartStatus', 'InputVoltage', 
                                            'OutputVoltage', 'OutputCurrent', 'OutputPower', 
                                            'Radius', 'QueryVectorRank']],
                                "Filtered Results with Radius and Query Vector Rank"
                            )
                        else:
                            st.warning("No active records found for this Part Number.")
                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")
    
    search_button = st.button("Search")
    
    if search_button and selected_model != "Select a Model":
        with st.spinner("Searching database..."):
            try:
                product_df = DF_AEI[
                    DF_AEI["PartNumber"].str.contains(selected_model, na=False, regex=False)
                ]
                
                if "PartStatus" in DF_AEI.columns:
                    product_df = product_df[product_df["PartStatus"] == "Active"]
                
                if not product_df.empty:
                    row_options = [
                        f"{format_column_name('PartNumber')}: {row['PartNumber']}, "
                        f"{format_column_name('OutputVoltage')}: {row['OutputVoltage']}, "
                        f"{format_column_name('OutputCurrent')}: {row['OutputCurrent']}, "
                        f"{format_column_name('OutputPower')}: {row['OutputPower']}"
                        for _, row in product_df.iterrows()
                    ]

                    selected_option = st.radio("Choose a Result for Cross-Reference Details", row_options)
                    selected_index = row_options.index(selected_option)
                    selected_row = product_df.iloc[selected_index]

                    query_vector = np.array([
                        selected_row['InputVoltage'],
                        selected_row['OutputVoltage'],
                        selected_row['OutputPower'],
                        selected_row['OutputCurrent']
                    ], dtype='float32').reshape(1, -1)

                    if input_voltage:
                        query_vector[0][available_specs.index("InputVoltage")] = get_numerical_values(input_voltage)
                    
                    k = 400
                    distances, indices = INDEX.search(query_vector, k)
                    
                    similar_items = DF.iloc[indices[0]].copy()
                    original_indices = indices[0]

                    if "PartStatus" in DF.columns:
                        mask = similar_items["PartStatus"] == "Active"
                        similar_items = similar_items[mask]
                        distances = distances[0][mask]
                        original_indices = original_indices[mask]
                    else:
                        distances = distances[0]
                    
                    similar_items['Radius'] = [
                        calculate_radius(query_vector, np.array([
                            row['InputVoltage'],
                            row['OutputVoltage'],
                            row['OutputPower'],
                            row['OutputCurrent']
                        ], dtype='float32'))
                        if pd.notnull(row['InputVoltage']) else None
                        for _, row in similar_items.iterrows()
                    ]
                    
                    similar_items['distance'] = distances
                    
                    similar_items = similar_items.sort_values(by='distance')

                    if Manufactureroptions:
                        similar_items = similar_items[similar_items["Manufacturer"].isin(Manufactureroptions)]

                    if ranked_specs:
                        similar_items = similar_items.sort_values(by=[spec[0] for spec in ranked_specs])

                    # Merge pricing data into similar_items
                    distinct_manufacturers = similar_items['Manufacturer'].unique()
                    manufacturer_prices = []

                    for manufacturer in distinct_manufacturers:
                        manufacturer_items = similar_items[similar_items["Manufacturer"] == manufacturer]
                        manufacturer_items = manufacturer_items.head(selected_number_records)
                        
                        digikey_price = digikey.getPrice(manufacturer_items.copy())
                        
                        if not digikey_price.empty:
                            # Select only PartNumber and available pricing columns
                            pricing_columns = [col for col in ['Unit Price', 'Break Quantity', 'Total Price'] if col in digikey_price.columns]
                            pricing_columns = ['PartNumber'] + pricing_columns
                            digikey_price = digikey_price[pricing_columns]
                            similar_items = similar_items.merge(
                                digikey_price,
                                on='PartNumber',
                                how='left'
                            )
                            manufacturer_prices.append(digikey_price)
                    
                    # New HTML output with pricing columns, distance, and scrolling
                    html_output = """
                    <style>
                        .table-container {
                            max-height: 300px;
                            overflow-y: scroll;
                            overflow-x: auto;
                            border: 1px solid #d1d5db;
                            border-radius: 8px;
                            background: #ffffff;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            margin-bottom: 20px;
                            display: block;
                            -webkit-overflow-scrolling: touch;
                            position: relative;
                        }
                        .table-container::-webkit-scrollbar {
                            width: 10px;
                            height: 10px;
                        }
                        .table-container::-webkit-scrollbar-track {
                            background: #f1f5f9;
                            border-radius: 4px;
                        }
                        .table-container::-webkit-scrollbar-thumb {
                            background: #6b7280;
                            border-radius: 4px;
                        }
                        .table-container::-webkit-scrollbar-thumb:hover {
                            background: #4b5563;
                        }
                        .table-container {
                            scrollbar-width: thin;
                            scrollbar-color: #6b7280 #f1f5f9;
                        }
                        table {
                            width: 100%;
                            min-width: 1300px; /* Ensure table is wide enough to trigger horizontal scroll */
                            border-collapse: collapse;
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            table-layout: auto;
                        }
                        th, td {
                            padding: 12px;
                            border-bottom: 1px solid #e5e7eb;
                            font-size: 13px;
                            color: #111827;
                            vertical-align: top;
                            white-space: nowrap; /* Prevent text wrapping */
                        }
                        th {
                            position: sticky;
                            top: 0;
                            background: #1f2937;
                            color: #ffffff;
                            font-weight: 600;
                            font-size: 14px;
                            text-align: left;
                            z-index: 10;
                            border-bottom: 2px solid #374151;
                        }
                        tr:nth-child(even) {
                            background-color: #f9fafb;
                        }
                        tr:hover {
                            background-color: #f3f4f6;
                        }
                        .divider {
                            border-top: 3px solid #4b5563;
                            margin: 30px 0;
                            border-radius: 2px;
                        }
                        .manufacturer-title {
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            font-size: 20px;
                            font-weight: 600;
                            color: #1f2937;
                            margin: 24px 0 12px;
                            padding-left: 6px;
                            border-left: 3px solid #2563eb;
                        }
                        .datasheet-link {
                            color: #2563eb;
                            text-decoration: none;
                            font-weight: 500;
                        }
                        .datasheet-link:hover {
                            color: #1e40af;
                            text-decoration: underline;
                        }
                        .error-message {
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            font-size: 14px;
                            color: #dc2626;
                            font-weight: 500;
                            padding: 12px;
                            background: #fef2f2;
                            border-radius: 6px;
                            margin: 12px 0;
                        }
                        .summary-container {
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            background: #f8fafc;
                            padding: 20px;
                            border-radius: 8px;
                            margin-top: 25px;
                            border-left: 4px solid #2563eb;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                        }
                        .summary-title {
                            font-size: 18px;
                            font-weight: 600;
                            color: #1e40af;
                            margin-bottom: 15px;
                        }
                        .summary-content {
                            line-height: 1.6;
                            color: #334155;
                        }
                        .summary-content ul {
                            padding-left: 20px;
                            margin-top: 10px;
                            margin-bottom: 10px;
                        }
                        .summary-content li {
                            margin-bottom: 8px;
                        }
                    </style>
                    """

                    df_by_manufacturer = [group.head(5) for _, group in similar_items.groupby('Manufacturer', sort=False)]
                    
                    for idx, group_df in enumerate(df_by_manufacturer):
                        manufacturer = group_df['Manufacturer'].iloc[0]
                        html_output += f'<h2 class="manufacturer-title">Manufacturer: {manufacturer}</h2>'
                        html_output += '<div class="table-container">'
                        html_output += '<table>'
                        html_output += '''
                        <thead>
                            <tr>
                                <th>Description</th>
                                <th>Part Number</th>
                                <th>Input Voltage</th>
                                <th>Output Voltage</th>
                                <th>Output Current</th>
                                <th>Output Power</th>
                                <th>Link</th>
                                <th>Radius</th>
                                <th>Distance</th>
                                <th>Unit Price</th>
                                <th>Break Quantity</th>
                                <th>Total Price</th>
                            </tr>
                        </thead>
                        <tbody>
                        '''
                        for _, row in group_df.iterrows():
                            link = row.get('DatasheetURL', '-')
                            datasheet_cell = f'<a class="datasheet-link" href="{link}" target="_blank">View Datasheet</a>' if (pd.notna(link) and isinstance(link, str) and link.startswith(('http://', 'https://'))) else '-'
                            unit_price = row.get('Unit Price', '-')
                            break_quantity = row.get('Break Quantity', '-')
                            total_price = row.get('Total Price', '-')
                            distance = row.get('distance', '-')
                            html_output += '<tr>'
                            html_output += f'<td>{row.get("PartDescription", "-")}</td>'
                            html_output += f'<td>{row["PartNumber"]}</td>'
                            html_output += f'<td>{row["InputVoltage"] if pd.notna(row["InputVoltage"]) else "-"}</td>'
                            html_output += f'<td>{row["OutputVoltage"]}</td>'
                            html_output += f'<td>{row["OutputCurrent"]}</td>'
                            html_output += f'<td>{row["OutputPower"]}</td>'
                            html_output += f'<td>{datasheet_cell}</td>'
                            html_output += f'<td>{row["Radius"]:.2f}</td>'
                            html_output += f'<td>{f"{distance:.2f}" if pd.notna(distance) else "-"}</td>'
                            html_output += f'<td>{unit_price if pd.notna(unit_price) else "-"}</td>'
                            html_output += f'<td>{break_quantity if pd.notna(break_quantity) else "-"}</td>'
                            html_output += f'<td>{total_price if pd.notna(total_price) else "-"}</td>'
                            html_output += '</tr>'
                        html_output += '</tbody></table></div>'
                        if idx < len(df_by_manufacturer) - 1:
                            html_output += '<div class="divider"></div>'
                    
                    with st.spinner(f"Generating conversational summary for {selected_model}..."):
                        summary = generate_conversational_summary(similar_items, selected_model)
                    
                    html_output += f"""
                    <div class="summary-container">
                        <div class="summary-title">AI Summary of Top Cross-Reference Parts for {selected_model}</div>
                        <div class="summary-content">{summary.replace('\n', '<br>')}</div>
                    </div>
                    """
                    
                    st.markdown(html_output, unsafe_allow_html=True)

                    # Preserve original pricing and comparison
                    for manufacturer in distinct_manufacturers:
                        manufacturer_items = similar_items[similar_items["Manufacturer"] == manufacturer]
                        manufacturer_items = manufacturer_items.head(selected_number_records)
                        
                        digikey_price = digikey.getPrice(manufacturer_items.copy())
                        display_table(digikey_price, f"{manufacturer} Pricing Data")

                        if not digikey_price.empty:
                            digikey_price.reset_index(drop=True, inplace=True)
                            manufacturer_prices.append(digikey_price)
                        else:
                            st.warning(f"No pricing data found for {manufacturer}.")
                    
                    if manufacturer_prices:
                        all_prices = pd.concat(manufacturer_prices).reset_index(drop=True)
                        all_prices['Total Price'] = all_prices['Total Price'].fillna(0)
                        
                        data = {
                            'Manufacturer': all_prices['Manufacturer'],
                            'Price': all_prices['Total Price']
                        }
                        df_prices = pd.DataFrame(data)
                        df_prices = df_prices[df_prices['Price'] != 0]

                        fig, ax = plt.subplots()
                        ax.barh(df_prices['Manufacturer'], df_prices['Price'])
                        ax.set_xlabel('Price')
                        ax.set_ylabel('Manufacturer')
                        ax.set_title('Price Comparison')
                        for i, v in enumerate(df_prices['Price']):
                            ax.text(v, i, " $"+str(v), color='black', va='center')
                        st.pyplot(fig)
                    else:
                        st.warning("No pricing data found for the selected manufacturers.")
                    
                else:
                    st.warning("No active records found for this Part Number.")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    elif search_button and selected_model == "Select a Model":
        st.warning("Please select a part from the dropdown before searching.")

if __name__ == "__main__":
    main()