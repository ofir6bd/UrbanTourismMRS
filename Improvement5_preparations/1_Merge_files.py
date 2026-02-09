import pandas as pd
import numpy as np
import os

def create_poi_descriptions(pois_file, output_file):
    # 1. Read POI data
    pois_df = pd.read_csv(pois_file, sep=';')
    
    print(f"POIs file shape: {pois_df.shape}")
    
    descriptions = []
    
    for idx, poi in pois_df.iterrows():
        poi_id = poi['id']
        poi_name = poi['name']
        
        # Clean up POI name
        clean_name = str(poi_name).replace('_', ' ').replace('(Roma)', '').strip()
        
        descriptions.append({
            'poi_id': poi_id,
            'clean_name': clean_name
        })
    
    # 2. Generate the ChatGPT Prompt
    chatgpt_file = output_file.replace('.csv', '_chatgpt_prompt.txt')
    with open(chatgpt_file, 'w', encoding='utf-8') as f:
        f.write("Act as an expert historian and architectural guide specialized in Rome.\n")
        f.write("For each POI listed below, please provide a detailed 2-sentence description.\n")
        f.write("Focus on: historical significance, architectural style, and tourist appeal.\n")
        f.write("IMPORTANT: Format your response as a Python dictionary named 'poi_texts' where the key is the ID and the value is the full name followed by your description.\n\n")
        
        f.write("poi_texts = {\n")
        for d in descriptions:
            # Requesting the full format: ID: "Name - Description"
            f.write(f'    {d["poi_id"]}: "{d["clean_name"]} – [ADD DESCRIPTION HERE]",\n')
        f.write("}\n")

    print(f"✓ Processed {len(descriptions)} POI entries.")
    print(f"✓ ChatGPT prompt generated: {chatgpt_file}")
    
    return descriptions

if __name__ == "__main__":
    # This will now only generate the .txt prompt file
    create_poi_descriptions("rome-pois.txt", "rome_poi_descriptions.csv")