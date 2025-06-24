#!/usr/bin/env python3
"""SWAPI starship module"""
import requests


def availableShips(passengerCount):
    """Method that returns the list of ships that can hold a given number
    of passengers:
        If no ship available, return an empty list"""
    valid_ships = []
    next_page = "https://swapi-api.hbtn.io/api/starships/"

    # Loop to iterate through all pages of starships
    while next_page:
        # Fetch current page
        response = requests.get(next_page)
        if response.status_code != 200:
            break

        page_data = response.json()

        # Check each starship on this page
        for starship in page_data['results']:
            passenger_info = starship['passengers']

            # Skip ships with no passenger data
            if passenger_info in ['n/a', 'unknown']:
                continue

            # Clean up passenger count and convert to integer
            passenger_num = int(passenger_info.replace(',', ''))

            # Add ship if it can hold enouch passengers
            if passenger_num >= passengerCount:
                valid_ships.append(starship['name'])

        # Move to next page
        next_page = page_data['next']

    return valid_ships
