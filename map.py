from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.exa import ExaTools
from dotenv import load_dotenv
import requests
import os

# Load environment variables (for API keys, etc.)
load_dotenv()

# Google Maps API setup
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # Store this in your .env file securely
BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Function to fetch important places like schools, hospitals, etc., without filtering by ratings
def fetch_important_places(location, radius=2500):
    place_types = [
        {"type": "school", "keyword": "school"},
        {"type": "hospital", "keyword": "hospital"},
        {"type": "police", "keyword": "police station"},
        {"type": "fire_station", "keyword": "fire station"},
        {"type": "pharmacy", "keyword": "pharmacy"},
    ]

    all_results = []
    for place in place_types:
        params = {
            "location": location,
            "radius": radius,
            "type": place["type"],
            "keyword": place["keyword"],
            "key": GOOGLE_MAPS_API_KEY,
        }
        response = requests.get(BASE_URL, params=params)
        results = response.json().get("results", [])

        for res in results:
            all_results.append({
                "name": res.get("name"),
                "type": place["type"],
                "rating": res.get("rating", "N/A"),
                "address": res.get("vicinity"),
                "total_ratings": res.get("user_ratings_total", "N/A"),
                "open_now": res.get("opening_hours", {}).get("open_now", "Unknown"),
            })

    return all_results

# Define Emergency Location Finder Agent
emergency_location_agent = Agent(
    name="Emergency Location Finder",
    tools=[
        ExaTools(),
    ],
    model=OpenAIChat(id="gpt-4"),
    description=(
        "You are Emergency Location Finder, an assistant that helps users identify critical locations near them like schools, hospitals, "
        "fire stations, police stations, and pharmacies that are important during emergencies such as accidents, tree falls, or health issues."
    ),
    instructions=[
        "Use location data to identify schools, hospitals, police stations, fire stations, and pharmacies near the user’s current location.",
        "Provide recommendations in a structured markdown table including name, type, rating (if available), address, total ratings, and open status.",
        "Offer safety tips, operational hours, and emergency contact suggestions if possible.",
    ],
    markdown=True,
)

# Function to generate markdown table for the important places
def recommend_important_places(location):
    results = fetch_important_places(location)
    if not results:
        return "No important places found near this location."

    table_header = "| Name | Type | Rating | Address | Total Ratings | Open Now |\n|---|---|---|---|---|---|\n"
    table_rows = "\n".join(
        f"| {res['name']} | {res['type'].capitalize()} | {res['rating']} | {res['address']} | {res['total_ratings']} | {res['open_now']} |"
        for res in results
    )
    return table_header + table_rows

# Example location: Coimbatore City (latitude, longitude)
if __name__ == "__main__":
    coimbatore_location = "11.0168,76.9558"
    recommendation_table = recommend_important_places(coimbatore_location)
    # Print the recommendation table
    print(recommendation_table)
