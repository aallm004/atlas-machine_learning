#!/usr/bin/env python3
"""SpaceX API upcoming launch script"""
import requests
from datetime import datetime


def get_first_upcoming_launch():
    """Script that displays the first launch with this information
        Name of the launch
        The date (in local time)
        The rocket name
        The name (with the locality) of the launchpad"""
    try:
        # Get upcoming launches
        launch_response = requests.get(
            "https://api.spacexdata.com/v4/launches/upcoming")
        if launch_response.status_code != 200:
            return

        launches = launch_response.json()
        if not launches:
            return

        # Sort by date_unix (first earliest)
        launches.sort(key=lambda x: x.get('date_unix', 0))
        first_launch = launches[0]

        # Get rocket data
        rocket_id = first_launch.get('rocket')
        rocket_response = \
            requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
        if rocket_response.status_code != 200:
            return

        rocket_data = rocket_response.json()
        rocket_name = rocket_data.get('name', 'Unknown')

        # Get launchpad data
        launchpad_id = first_launch.get('launchpad')
        launchpad_response = requests.get(
            f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
        if launchpad_response.status_code != 200:
            return

        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get('name', 'Unknown')
        launchpad_locality = launchpad_data.get('locality', 'Unknown')

        # Extract launch info
        launch_name = first_launch.get('name', 'Unknown')
        launch_date = first_launch.get('date_local', 'Unknown')

        # Format output
        print(f"{launch_name} ({launch_date}) {rocket_name} - \
              {launchpad_name} ({launchpad_locality})")

    except requests.RequestException:
        pass
    except Exception:
        pass


if __name__ == '__main__':
    get_first_upcoming_launch()
