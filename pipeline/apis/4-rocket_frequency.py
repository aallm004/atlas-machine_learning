#!/usr/bin/env python3
"""SpaceX API rocket launch frequency script"""
import requests


def get_rocket_launch_frequency():
    """Script that displays the number of launches per rocket"""
    try:
        # Get launches
        launches_response = requests.get("https://api.spacexdata.com/v4/launches")
        if launches_response.status_code != 200:
            return
        
        launches = launches_response.json()

        # Count rocket IDs
        rocket_counts = {}
        for launch in launches:
            rocket_id = launch.get('rocket')
            if rocket_id:
                rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1

        # Rocket names
        rocket_names = {}
        for rocket_id in rocket_counts:
            rocket_response = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
            if rocket_response.status_code == 200:
                rocket_data = rocket_response.json()
                rocket_names[rocket_id] = rocket_data.get('name', 'Unknown')

        # Create list of (name, count) tuples
        rocket_stats = []
        for rocket_id, count in rocket_counts.items():
            name = rocket_names.get(rocket_id, 'Unknown')
            rocket_stats.append((name, count))

        # Sort by count (descending), then by name (ascending)
        rocket_stats.sort(key=lambda x: (-x[1], x[0]))

        # Print results
        for name, count in rocket_stats:
            print(f"{name}: {count}")

    except requests.RequestException:
        pass
    except Exception:
        
if __name__ == '__main__':
    get_rocket_launch_frequency()
