import requests

# Get game details
game_id = 3498  # GTA V
url = f"https://api.rawg.io/api/games/{game_id}"
params = {
    "key": "22f806f97eaf4916b525c2b787b52fc1"
}

response = requests.get(url, params=params)
data = response.json()

# Print all keys in the response
print("Available fields:", list(data.keys()))
print("\nDescription field:", data.get("description", "Not found")[:200])
