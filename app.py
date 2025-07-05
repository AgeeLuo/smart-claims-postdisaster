## Import Libraries
import os
import re
import math
import json
import random
from datetime import datetime, timedelta

# ---
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import st_folium

# ---
import ee
import geemap
import geemap.foliumap as geemap_folium

# ---
import ollama


# --------- CONFIG ---------


st.set_page_config(
    page_title="🏠 Smart Insurance Claims After Disasters", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)           

st.markdown(
    """
    <div style='background-color: #e3f2fd; padding: 30px; border-radius: 10px;'>
        <h1 style='text-align: center; margin-bottom: 10px;'>Smart Insurance Claims After Disasters</h1>
        <p style='text-align: center; font-size: 18px; color: gray;'>
            A smart tool powered by <strong>GenAI</strong> and <strong>Satellite Imagery</strong>, 
            designed to accelerate insurance claims by identifying impacted policyholders following a disaster.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Initialize session state
if 'selected_event' not in st.session_state:
    st.session_state.selected_event = None
if 'event_data' not in st.session_state:
    st.session_state.event_data = {}
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

headers = {"User-Agent": "Mozilla/5.0"}
base_url = "https://reliefweb.int"

# Load area border data: GAUL shapefile  
@st.cache_resource
def load_shapefile():
    try:
        return gpd.read_file("GAUL_2024_L1/GAUL_2024_L1.shp")
    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        return None

gdf = load_shapefile()
if gdf is not None:
    country_col = "gaul0_name"
    admin1_col = "gaul1_name"



# --------- FUNCTIONS ---------
@st.cache_data(ttl=3600)
# Fetch the most recent flood events from ReliefWeb
def fetch_flood_events():
    try:
        url = f"{base_url}/disasters"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        events = soup.find_all("h3", class_="rw-river-article__title")
        now = datetime.now()
        three_months_ago = now - relativedelta(months=3)
        seen = set()
        data = []
        
        for event in events:
            a_tag = event.find("a")
            if not a_tag:
                continue
            title = a_tag.text.strip()
            link = a_tag.get("href", "")
            
            if re.search(r'\bFloods\b', title, re.IGNORECASE):
                match = re.match(r"^(.*?): Floods.*- (\w+ \d{4})", title)
                if match:
                    country = match.group(1)
                    date_str = match.group(2)
                    try:
                        event_date = datetime.strptime(date_str, "%b %Y")
                    except ValueError:
                        continue
                    
                    if event_date >= three_months_ago:
                        formatted_date = event_date.strftime("%Y/%m")
                        key = (country, formatted_date)
                        if key not in seen:
                            seen.add(key)
                            # Ensure link is absolute
                            if link.startswith('/'):
                                link = base_url + link
                            data.append({
                                "Country": country,
                                "Date": formatted_date,
                                "Link": link
                            })
        return data
    except Exception as e:
        st.error(f"Error fetching flood events: {e}")
        return []

# Scrape flood details
def scrape_disaster_description(link):
    try:
        response = requests.get(link, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        section = soup.find("section", id="overview")
        if section:
            content_div = section.find("div", class_="rw-entity-text__content")
            if content_div:
                return content_div.get_text(separator="\n", strip=True)
        return "No disaster description section found."
    except Exception as e:
        return f"Error scraping content: {e}"

# Use genai to search for the flood area from flood details
def query_ollama_llm(text):
    prompt = f"""You are an information extractor. Given the following disaster description, extract:
- Country
- Province or state (most precise location mentioned)
- Exact date of the disaster (or date range if available)

Respond in JSON format like:
{{"country": "...", "province_or_state": "...", "date": "YYYY-MM-DD"}}.

Description:
{text}
"""
    try:
        response = ollama.generate(model="gemma3", prompt=prompt)
        raw = response['response'].strip()
        if raw.startswith("```json"):
            raw = raw.strip("` \n")
            raw = re.sub(r"^json\s*", "", raw, flags=re.IGNORECASE)
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e), "raw_response": response.get("response", "")}

def match_admin1_with_llm(country, province_or_state):
    if gdf is None:
        return {"error": "Shapefile not loaded"}
    
    filtered = gdf[gdf[country_col].str.lower() == country.lower()]
    if filtered.empty:
        return {"error": f"No match found for country: {country}"}
    
    admin_list = sorted(filtered[admin1_col].dropna().unique())
    prompt = f"""You are given a province or state name and a list of administrative regions (GAUL Level 1).
Your job is to pick the most likely matching region from the list.
Return a JSON object in this format:
{{"matched_admin1": "..."}}

Country: {country}
Province or State: {province_or_state}

GAUL Level 1 List:
{chr(10).join(f"- {name}" for name in admin_list)}
"""
    try:
        response = ollama.generate(model="gemma3", prompt=prompt)
        raw = response['response'].strip()
        if raw.startswith("```json"):
            raw = raw.strip("` \n")
            raw = re.sub(r"^json\s*", "", raw, flags=re.IGNORECASE)
        parsed = json.loads(raw)
        return {
            "country": country,
            "province_or_state": province_or_state,
            "matched_admin1": parsed.get("matched_admin1", None)
        }
    except Exception as e:
        return {"error": str(e), "raw_response": response.get("response", "")}

# Initiate Earth Engine
def initialize_earth_engine():
    try:
        
        # Initialize EE if not already done
        if not hasattr(st.session_state, 'ee_initialized'):
            try:
                # ee.Initialize(project='rapid-stream-451101-b9')  # Put your project ID in .env file
                from dotenv import load_dotenv
                load_dotenv()
                project_id = os.getenv('GOOGLE_EARTH_ENGINE_PROJECT_ID')
                if not project_id:
                    st.error("Please set GOOGLE_EARTH_ENGINE_PROJECT_ID environment variable")
                    return False
                ee.Initialize(project=project_id)
                st.session_state.ee_initialized = True
            except Exception as e:
                st.error(f"Earth Engine initialization failed: {e}")
                return False
        return True
    except ImportError as e:
        st.error(f"Required libraries not available: {e}")
        return False

# Create fictional customers: names
def fake_name(index):
    first_names = ['Ali', 'Sara', 'John', 'Ayesha', 'Mohammed', 'Anna', 'David', 'Lina', 'Ravi', 'Fatima']
    last_names = ['Khan', 'Smith', 'Lee', 'Patel', 'Ahmed', 'Chen', 'Singh', 'Zhao', 'Iqbal', 'Tan']
    return f"{random.choice(first_names)} {random.choice(last_names)}_{index+1}"

# Create fictional customers: loction 
def generate_people_feature_collection(location_name, center_lat, center_lon, radius_km=10, num=100):
    if not initialize_earth_engine():
        return False
    
    try:

        earth_radius_km = 6371.0
        features = []

        for i in range(num):
            angle = random.uniform(0, 2 * math.pi)
            r = radius_km * math.sqrt(random.uniform(0, 1))
            delta_lat = (r / earth_radius_km) * (180 / math.pi)
            delta_lon = delta_lat / math.cos(center_lat * math.pi / 180)
            lat = center_lat + delta_lat * math.sin(angle)
            lon = center_lon + delta_lon * math.cos(angle)

            point = ee.Geometry.Point([lon, lat])
            props = {
                'issued_person_name': fake_name(i),
                'location_name': location_name
            }
            features.append(ee.Feature(point, props))

        return ee.FeatureCollection(features)

    except Exception as e:
        st.error(f"🌐 Failed to generate customer data: {e}")
        return False

# Locate customers impacted by flood
def get_home_near_flood(homes, flood_extent, buffer_distance_m=10000, flood_buffer_m=200, scale=30):
    """
    Returns a FeatureCollection of homes that are near flood areas.
    
    Parameters:
        homes (ee.FeatureCollection): Home/household data points.
        flood_extent (ee.Image): Flood extent image (binary image, 1 represents flood).
        buffer_distance_m (int): Analysis area radius, default 10 kilometers.
        flood_buffer_m (int): Flood area buffer distance, default 200 meters.
        scale (int): Image resolution (default 30 meters).
    
    Returns:
        ee.FeatureCollection: FeatureCollection of homes near flood areas that can be added to a map.
    """
    try:
        # Define analysis area of interest
        aoi = homes.geometry().buffer(buffer_distance_m)

        # Convert flood extent to vector format
        flood_vector = flood_extent.reduceToVectors(
            geometry=aoi,
            scale=scale,
            geometryType='polygon',
            labelProperty='flood',
            reducer=ee.Reducer.countEvery(),
            bestEffort=True,
            maxPixels=1e8
        )

        # Apply buffer to flood polygons
        flood_buffer = flood_vector.map(lambda f: f.buffer(flood_buffer_m))

        # Filter homes that are near flood areas
        homes_near_flood = homes.filterBounds(flood_buffer.geometry())

        # Retrieve homes_near_flood locally to get coordinates
        fc_info = homes_near_flood.getInfo()

        home_locations = []
        for idx, feature in enumerate(fc_info['features']):
            lon, lat = feature['geometry']['coordinates']
            props = feature['properties']
            home_locations.append({
                "name": f"Home {idx + 1}",
                "lat": lat,
                "lon": lon,
                "issued_person_name": props.get("issued_person_name", ""),
                "location_name": props.get("location_name", "")
            })

        # Reconstruct FeatureCollection

        home_features = [
            ee.Feature(
                ee.Geometry.Point(h['lon'], h['lat']),
                {
                    'name': h['name'],
                    'issued_person_name': h['issued_person_name'],
                    'location_name': h['location_name']
                }
            )
            for h in home_locations
        ]
        flooded_home_fc = ee.FeatureCollection(home_features)

        return flooded_home_fc

    except Exception as e:
        print(f"⚠️ Failed to create home layer: {e}")
        return False

# Show the map: flood extent and impacted customers
def show_flood_map(matched):
    if not initialize_earth_engine():
        return False
    
    try: 
        country_name = matched["country"]
        admin1_name = matched["matched_admin1"]
        event_date = datetime.strptime(matched["date"], "%Y-%m-%d")
        
        flood_start = (event_date - timedelta(days=30)).strftime("%Y-%m-%d")
        flood_end = (event_date + timedelta(days=30)).strftime("%Y-%m-%d")
        pre_start = (event_date - timedelta(days=395)).strftime("%Y-%m-%d")
        pre_end = (event_date - timedelta(days=31)).strftime("%Y-%m-%d")
        
        gaul = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level1')
        admin1 = gaul.filter(
            ee.Filter.And(
                ee.Filter.eq("ADM0_NAME", country_name),
                ee.Filter.eq("ADM1_NAME", admin1_name)
            )
        )
        region = admin1.first().geometry()
        
        def get_sar_composite(start, end):
            collection = (
                ee.ImageCollection('COPERNICUS/S1_GRD')
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                .filterDate(start, end)
                .filterBounds(region)
                .select('VV')
                .map(lambda img: img.focal_median(100, 'circle', 'meters'))
            )
            return collection.reduce(ee.Reducer.percentile([20])).clip(region)
        
        sar_before = get_sar_composite(pre_start, pre_end)
        sar_after = get_sar_composite(flood_start, flood_end)
        
        threshold = -18
        water_before = sar_before.lt(threshold).selfMask()
        water_after = sar_after.lt(threshold).selfMask()
        flood_extent = water_after.unmask().subtract(water_before.unmask()).gt(0).selfMask()
        
        coords = region.centroid().coordinates().getInfo()
        lon, lat = coords[0], coords[1]
        
        # Create folium map
        Map = folium.Map(location=[lat, lon], zoom_start=8)

        # Water layers: Use light blue and blue with higher transparency
        Map.add_child(geemap_folium.ee_tile_layer(
            water_before, {'palette': ['#ADD8E6'], 'opacity': 0.1}, 'Before Flood Water (Light Blue)'
        ))

        Map.add_child(geemap_folium.ee_tile_layer(
            water_after, {'palette': ['#0000FF'], 'opacity': 0.2}, 'After Flood Water (Blue)'
        ))

        # flood extent：Purple
        Map.add_child(geemap_folium.ee_tile_layer(
            flood_extent, {'palette': ["#C300FF"], 'opacity': 1}, 'Flood Extent'
        ))

        # Border style
        style = {'color': 'black', 'fillColor': '00000000', 'width': 2}
        Map.add_child(geemap_folium.ee_tile_layer(admin1.style(**style), {}, f"{admin1_name} Boundary"))

        # Fictional customer data
        issued_fc = generate_people_feature_collection(country_name, lat, lon, radius_km=10, num=20)
        flooded_fc = get_home_near_flood(
            homes=issued_fc,
            flood_extent=flood_extent,
            buffer_distance_m=10000,
            flood_buffer_m=200,
            scale=30
        )

        # Customer data viz
        def add_emoji_marker(fc, icon_emoji, icon_color):
            features = fc.getInfo()['features']
            for f in features:
                coords = f['geometry']['coordinates']
                lon, lat = coords
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=f"""<div style="font-size: 32px;">{icon_emoji}</div>"""
                    ),
                    tooltip=f['properties'].get('name', 'Customer')
                ).add_to(Map)

        # Display 😳as unaffeted customer
        add_emoji_marker(issued_fc, '😳', 'yellow')

        # Display 🤢as impacted customer
        add_emoji_marker(flooded_fc, '🤢', 'blue')


        
        # Remove original legend_html and Map.get_root().html.add_child(folium.Element(legend_html))
        st.markdown("---")
        st.subheader("🗺️ Locate Impacted Customer")
        st.markdown(
    "Utilize **geemap** and **Sentinel-1** data (from Google Earth Engine) to visualize flood extent "
    "and locate impacted (fictional) customers."
)
        
        # === Use Streamlit columns to display map and legend ===
        col1, col2 = st.columns([4, 1])  # Map takes 4/5 width, legend takes 1/5 width
        
        with col1:
            # Display map and return interaction data
            map_data = st_folium(Map, height=1000, width=None, returned_objects=["last_clicked"])
            
            # If user clicked on map
            if map_data and map_data.get("last_clicked"):
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lon = map_data["last_clicked"]["lng"]
                st.info(f"🧭 You clicked at: Latitude = {clicked_lat:.6f}, Longitude = {clicked_lon:.6f}")
        
        with col2:
            # Display legend
            st.markdown("##### 📘 Legend")
            
            # Use HTML to create better-looking legend
            
            legend_html = """
            <div style="font-size: 14px; line-height: 1.8;">
                <div style="margin: 8px 0;">
                    <span style="display: inline-block; width: 20px; height: 15px; background-color: #ADD8E6; margin-right: 8px; border: 1px solid #ccc; opacity: 1;"></span>
                    <span>Before Flood Water</span>
                </div>
                <div style="margin: 8px 0;">
                    <span style="display: inline-block; width: 20px; height: 15px; background-color: #0000FF; margin-right: 8px; border: 1px solid #ccc; opacity: 1;"></span>
                    <span>After Flood Water</span>
                </div>
                <div style="margin: 8px 0;">
                    <span style="display: inline-block; width: 20px; height: 15px; background-color: #C300FF; margin-right: 8px; border: 1px solid #ccc; opacity: 1;"></span>
                    <span>Flood Extent</span>
                </div>
                <div style="margin: 8px 0;">
                    <span style="font-size: 18px; margin-right: 8px;">😳</span>
                    <span>Customer (Unaffected)</span>
                </div>
                <div style="margin: 8px 0;">
                    <span style="font-size: 18px; margin-right: 8px;">🤢</span>
                    <span>Impacted Customer (Flooded)</span>
                </div>
                <div style="margin: 8px 0;">
                    <span style="display: inline-block; width: 20px; height: 2px; background-color: black; margin-right: 8px;"></span>
                    <span>Boundary</span>
                </div>
            </div>
            """
            
            st.markdown(legend_html, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("##### 📊 Map Info")
            st.markdown(f"**Country/Region:** {country_name}/{admin1_name}")
            st.markdown(f"**Event Date:** {matched['date']}")
            st.markdown(f"**Center:** {lat:.4f}, {lon:.4f}")
            st.markdown("---")
            st.markdown("##### 🤢 Impacted Customers")

            def ee_featurecollection_to_df(fc, properties):
                """ 將 FeatureCollection 轉為 pandas DataFrame，取出指定屬性 """
                import pandas as pd
                geojson = fc.getInfo()
                rows = []
                for feature in geojson['features']:
                    row = {prop: feature['properties'].get(prop, '') for prop in properties}
                    rows.append(row)
                return pd.DataFrame(rows)

            impacted_df = ee_featurecollection_to_df(flooded_fc, ['issued_person_name', 'location_name'])

            if not impacted_df.empty:
                st.dataframe(impacted_df)
            else:
                st.info("🚫 No impacted customers found.")

        
        return True
        
    except Exception as e:
        st.error(f"🌐 Failed to generate flood map: {e}")
        return False


def display_results():
    # st.info("📄 Event Details!")
    
    # Display extracted details
    with st.expander("📍 Result from ReliefWeb", expanded=False):
        st.json(st.session_state.event_data['llm_result'])
    
    # Display matched region
    with st.expander("🗺️ Matched GAUL Region", expanded=False):
        st.json(st.session_state.event_data['matched'])
    
    # Generate map
    if 'error' not in st.session_state.event_data['matched']:
        with st.spinner("Generating flood extent map..."):
            map_success = show_flood_map(st.session_state.event_data['matched'])
            if map_success:
                st.success("🗺️ Map generated successfully!")
    else:
        st.error(f"Cannot generate map: {st.session_state.event_data['matched']['error']}")
    


# --------- APP LOGIC ---------
def main():
    # Fetch events
    with st.spinner("Loading flood events..."):
        events = fetch_flood_events()

    if not events:
        st.info("No recent flood events found.")
        return

    # Display event list
    st.subheader("🌊 Choose Target Flood Event")
    st.markdown(
    "Please choose one flood event from [ReliefWeb](https://reliefweb.int) as your target."
)
    
    # Create a more robust button layout
    for i, event in enumerate(events):
        event_key = f"{event['Country']}_{event['Date']}"
        button_label = f"🌊 {event['Country']} - {event['Date']}"
        
        # Use columns for better layout
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            if st.button(button_label, key=f"btn_{event_key}_{i}", use_container_width=True):
                # Clear previous data when selecting new event
                st.session_state.selected_event = event
                st.session_state.processing_complete = False
                st.session_state.event_data = {}
                st.rerun()
    
    # Process selected event
    if st.session_state.selected_event:
        process_selected_event()



def process_selected_event():
    selected_event = st.session_state.selected_event
    st.divider()
    st.subheader(f"📌 Locate Flood Area: {selected_event['Country']} - {selected_event['Date']}")
    st.markdown(
    "The flood area will be automatically identified by analyzing the event details on "
    "[ReliefWeb](https://reliefweb.int) and matching them with the "
    "[FAO GAUL border dataset](https://data.apps.fao.org/catalog/iso/34f97afc-6218-459a-971d-5af1162d318a)."
)

    
    # Process event if not completed
    if not st.session_state.processing_complete:
        status_container = st.empty()
        
        with st.spinner(f"Processing {selected_event['Country']} flood event..."):
            # Step 1: Scrape flood description
            status_container.info("Description scraped successfully!")
            description = scrape_disaster_description(selected_event["Link"])
            
            if "Error" in description or "No disaster" in description:
                st.error(description)
                return
            
            st.session_state.event_data['description'] = description
            
            # Step 2: Use LLM to extract flood area
            status_container.info("Information extracted successfully!")
            result = query_ollama_llm(description)
            st.session_state.event_data['llm_result'] = result
            
            if "error" in result:
                st.error(f"LLM Error: {result['error']}")
                return
            
            # Step 3: Match flood area with a GAUL region
            status_container.info("Region matched successfully!")
            matched = match_admin1_with_llm(
                country=result.get("country", ""),
                province_or_state=result.get("province_or_state", "")
            )
            matched["date"] = result.get("date", "")
            st.session_state.event_data['matched'] = matched
            st.session_state.processing_complete = True
            
            # Display event processing completion
            status_container.success("✅ Event processing completed!")
    
    # Display results
    if st.session_state.processing_complete and st.session_state.event_data:
        display_results()



# Run the app
if __name__ == "__main__":
    main()

