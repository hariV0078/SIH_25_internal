from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.exa import ExaTools
from dotenv import load_dotenv
import requests
import os
import math

# Load environment variables (for API keys, etc.)
load_dotenv()

# Google Maps API setup
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # Store this in your .env file securely
BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
FIXED_RADIUS = 1000  # Fixed radius in meters

# Problem severity mapping (higher score = higher priority)
PROBLEM_SEVERITY = {
    "fallen tree": 9,           # High - blocks roads, emergency access
    "water leakage": 8,         # High - can cause structural damage, health issues
    "broken streetlight": 7,    # Medium-High - safety concern, especially at night
    "pothole": 6,              # Medium - vehicle damage, safety
    "garbage dump": 5,         # Medium - health and sanitation
    "damaged road": 6,         # Medium - traffic disruption
    "traffic signal": 8,       # High - traffic safety
    "power outage": 9,         # High - affects everything
    "sewage overflow": 8,      # High - health hazard
    "broken sidewalk": 4,      # Medium-Low - pedestrian safety
    "vandalism": 3,            # Low - aesthetic issue
    "noise pollution": 2,      # Low - quality of life
}

# Place type importance weights (higher = more important to be nearby)
PLACE_IMPORTANCE = {
    "hospital": 10,
    "school": 8,
    "fire_station": 9,
    "police": 9,
    "pharmacy": 7,
}

def calculate_distance(lat1, lng1, lat2, lng2):
    """
    Calculate distance between two coordinates using Haversine formula
    Returns distance in meters
    """
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = (math.sin(delta_lat/2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lng/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

# Function to fetch important places like schools, hospitals, etc., without filtering by ratings
def fetch_important_places(location, radius=FIXED_RADIUS):
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
                "lat": res.get("geometry", {}).get("location", {}).get("lat"),
                "lng": res.get("geometry", {}).get("location", {}).get("lng"),
            })

    return all_results

def prioritize_problems(problems):
    """
    Prioritize problems based on:
    1. Problem severity
    2. Proximity to important places (hospitals, schools, etc.)
    3. Number of important places nearby
    
    Args:
        problems: List of dictionaries with 'lat', 'lng', and 'problem' keys
    
    Returns:
        List of problems with priority scores and reasons, sorted by priority (highest first)
    """
    prioritized = []
    
    for problem in problems:
        lat = problem['lat']
        lng = problem['lng']
        problem_type = problem['problem'].lower()
        location = f"{lat},{lng}"
        
        # Get base severity score
        severity_score = PROBLEM_SEVERITY.get(problem_type, 5)  # Default to medium priority
        
        # Get nearby important places
        nearby_places = fetch_important_places(location, FIXED_RADIUS)
        
        # Calculate proximity and importance scores
        proximity_score = 0
        important_places_nearby = []
        place_counts = {"hospital": 0, "school": 0, "fire_station": 0, "police": 0, "pharmacy": 0}
        
        for place in nearby_places:
            if place['lat'] and place['lng']:
                distance = calculate_distance(lat, lng, place['lat'], place['lng'])
                place_type = place['type']
                place_counts[place_type] += 1
                
                # Calculate proximity bonus (closer = higher score)
                # Max bonus of 3 points for places within 500m, decreasing with distance
                if distance <= 500:
                    distance_bonus = 3
                elif distance <= 1000:
                    distance_bonus = 2
                elif distance <= 2000:
                    distance_bonus = 1
                else:
                    distance_bonus = 0.5
                
                place_importance = PLACE_IMPORTANCE.get(place_type, 5)
                proximity_score += (place_importance * distance_bonus) / 10
                
                if distance <= 1000:  # Only count very close places in the description
                    important_places_nearby.append({
                        "name": place['name'],
                        "type": place_type,
                        "distance": round(distance)
                    })
        
        # Total priority score
        total_score = severity_score + proximity_score
        
        # Generate reason for prioritization
        reasons = []
        
        # Add severity reason
        if severity_score >= 8:
            reasons.append(f"High severity issue ({problem_type}) requiring immediate attention")
        elif severity_score >= 6:
            reasons.append(f"Medium-high priority ({problem_type}) affecting safety/infrastructure")
        else:
            reasons.append(f"Standard priority issue ({problem_type})")
        
        # Add proximity reasons
        if place_counts['hospital'] > 0:
            reasons.append(f"{place_counts['hospital']} hospital(s) nearby - affects emergency access")
        if place_counts['school'] > 0:
            reasons.append(f"{place_counts['school']} school(s) nearby - affects student safety")
        if place_counts['fire_station'] > 0:
            reasons.append(f"{place_counts['fire_station']} fire station(s) nearby - affects emergency response")
        if place_counts['police'] > 0:
            reasons.append(f"{place_counts['police']} police station(s) nearby - affects public safety operations")
        if place_counts['pharmacy'] > 0:
            reasons.append(f"{place_counts['pharmacy']} pharmacy(ies) nearby - affects medical access")
        
        if not any(place_counts.values()):
            reasons.append("No critical infrastructure nearby - lower priority")
        
        # Create prioritized problem entry
        prioritized_problem = {
            "lat": lat,
            "lng": lng,
            "problem": problem['problem'],
            "priority_score": round(total_score, 2),
            "severity_score": severity_score,
            "proximity_score": round(proximity_score, 2),
            "nearby_important_places": len(nearby_places),
            "critical_places_within_1km": len(important_places_nearby),
            "reason": "; ".join(reasons),
            "nearby_places_detail": important_places_nearby[:2],  # Limit to top 2 for brevity
            "place_counts": place_counts
        }
        
        prioritized.append(prioritized_problem)
    
    # Sort by priority score (highest first)
    prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Add rank to each problem
    for i, problem in enumerate(prioritized, 1):
        problem['rank'] = i
        
        # Add urgency level based on score
        if problem['priority_score'] >= 15:
            problem['urgency'] = "CRITICAL"
        elif problem['priority_score'] >= 12:
            problem['urgency'] = "HIGH"
        elif problem['priority_score'] >= 9:
            problem['urgency'] = "MEDIUM"
        else:
            problem['urgency'] = "LOW"
    
    return prioritized

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
        "Use location data to identify schools, hospitals, police stations, fire stations, and pharmacies near the user's current location.",
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
    # Test with the provided sample data
    sample_problems = [
        {
            "lat": 12.9716,
            "lng": 77.5946,
            "problem": "fallen tree"
        },
        {
            "lat": 13.0827,
            "lng": 80.2707,
            "problem": "pothole"
        },
        {
            "lat": 17.3850,
            "lng": 78.4867,
            "problem": "water leakage"
        },
        {
            "lat": 28.7041,
            "lng": 77.1025,
            "problem": "broken streetlight"
        },
        {
            "lat": 19.0760,
            "lng": 72.8777,
            "problem": "garbage dump"
        }
    ]
    
    prioritized = prioritize_problems(sample_problems)
    
    print("PRIORITIZED PROBLEMS:")
    print("=" * 50)
    for problem in prioritized:
        print(f"Rank {problem['rank']}: {problem['problem'].title()}")
        print(f"Location: {problem['lat']}, {problem['lng']}")
        print(f"Priority Score: {problem['priority_score']}")
        print(f"Urgency: {problem['urgency']}")
        print(f"Reason: {problem['reason']}")
        print(f"Nearby Important Places: {problem['nearby_important_places']}")
        print("-" * 50)
