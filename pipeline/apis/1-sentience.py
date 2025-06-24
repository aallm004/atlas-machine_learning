#!/usr/bin/env python3
"""SWAPI sentient species module"""
import requests


def sentientPlanets():
    """Method that returns the list of names of the home planets of all
    sentient species
        sentient type is either in the classification or designation attributes
    """
    planet_names = []
    next_page = "https://swapi-api.hbtn.io/api/species/"

    # Loop through pages of species
    while next_page:
        # Get current page
        response = requests.get(next_page)
        if response.status_code != 200:
            break

        page_data = response.json()

        # Check each species on page
        for species in page_data['results', []]:
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()

            # Check if species is sentient
            is_sentient = ('sentient' in classification or
                           'sentient' in designation or
                           classification == 'sentient' or
                           designation == 'sentient')

            # Check if species is sentient
            if is_sentient:
                homeworld_url = species.get('homeworld')

                # Skip if no homeworld URL
                if not homeworld_url or homeworld_url == 'null':
                    if 'unknown' not in planet_names:
                            planet_names.append('unknown')
                    continue

                # Get planet data
                try:
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        planet_name = planet_data.get('name')

                        # Add planet name if it was able to be retrieved
                        if planet_name and planet_name not in planet_names:
                            planet_names.append(planet_name)

                    else:
                        # If unable to fetch planet data, add as unknown
                        if 'unknown' not in planet_names:
                            planet_names.append('unknown')
                except:
                    # Network error or other
                    if 'unknown' not in planet_names:
                        planet_names.append('unknown')
            # Go to next page
            next_page = page_data.get('next')

        return planet_names

