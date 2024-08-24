import streamlit as st
import requests

# Set the backend URL
backend_url = "http://127.0.0.1:8000/predict"

# Streamlit app title
st.title("Mushroom Classification Prediction")

# Create three columns
col1, col2, col3 = st.columns(3)

# First column dropdowns
with col1:
    cap_shape = st.selectbox('Cap Shape', ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'])
    cap_surface = st.selectbox('Cap Surface', ['fibrous', 'grooves', 'scaly', 'smooth'])
    cap_color = st.selectbox('Cap Color', ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'])
    bruises = st.selectbox('Bruises', ['bruises', 'no'])
    odor = st.selectbox('Odor', ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'])
    population = st.selectbox('Population', ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'])
    habitat = st.selectbox('Habitat', ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'])

# Second column dropdowns
with col2:
    gill_attachment = st.selectbox('Gill Attachment', ['attached', 'free'])
    gill_spacing = st.selectbox('Gill Spacing', ['close', 'crowded'])
    gill_size = st.selectbox('Gill Size', ['broad', 'narrow'])
    gill_color = st.selectbox('Gill Color', ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'])
    stalk_shape = st.selectbox('Stalk Shape', ['enlarging', 'tapering'])
    stalk_root = st.selectbox('Stalk Root', ['bulbous', 'club', 'equal', 'rooted', 'missing'])
    spore_print_color = st.selectbox('Spore Print Color', ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'])

# Third column dropdowns
with col3:
    stalk_surface_above_ring = st.selectbox('Stalk Surface Above Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
    stalk_surface_below_ring = st.selectbox('Stalk Surface Below Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
    stalk_color_above_ring = st.selectbox('Stalk Color Above Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
    stalk_color_below_ring = st.selectbox('Stalk Color Below Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
    veil_type = st.selectbox('Veil Type', ['partial', 'universal'])
    veil_color = st.selectbox('Veil Color', ['brown', 'orange', 'white', 'yellow'])
    ring_number = st.selectbox('Ring Number', ['none', 'one', 'two'])
    ring_type = st.selectbox('Ring Type', ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'])

# When the user clicks the button
if st.button('Predict'):
    # Prepare the data as a dictionary
    params = {
        'cap_shape': cap_shape,
        'cap_surface': cap_surface,
        'cap_color': cap_color,
        'bruises': bruises,
        'odor': odor,
        'gill_attachment': gill_attachment,
        'gill_spacing': gill_spacing,
        'gill_size': gill_size,
        'gill_color': gill_color,
        'stalk_shape': stalk_shape,
        'stalk_root': stalk_root,
        'stalk_surface_above_ring': stalk_surface_above_ring,
        'stalk_surface_below_ring': stalk_surface_below_ring,
        'stalk_color_above_ring': stalk_color_above_ring,
        'stalk_color_below_ring': stalk_color_below_ring,
        'veil_type': veil_type,
        'veil_color': veil_color,
        'ring_number': ring_number,
        'ring_type': ring_type,
        'spore_print_color': spore_print_color,
        'population': population,
        'habitat': habitat
    }

    # Send a GET request to the Flask backend
    response = requests.get(backend_url, params=params)

    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Confidence Score: {result['score']:.2f}")
    else:
        st.error("Failed to get a response from the backend")
