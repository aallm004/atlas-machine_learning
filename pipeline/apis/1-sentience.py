#!/usr/bin/env python3
"""SWAPI sentient species module"""
import requests


def sentientPlanets():
    """Method that returns the list of names of the home planets of all
    sentient species
        sentient type is either in the classification or designation attributes
    """
    planets = []
    page = 1
    base_url = "https://swapi-api.hbtn.io/api/species/"

    while True:
        # Build URL with page parameter
        url = f"{base_url}?page={page}" if page > 1 else base_url

        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                break
        except requests.RequestException:
            break

        data = resp.json()
        species_list = data['results']

        # Process each species in current batch
        for item in species_list:
            class_type = item.get('classification', '')
            design_type = item.get('designation', '')

            if 'sentient' in class_type.lower() or \
               'sentient' in design_type.lower():
                world_link = item.get('homeworld')

                if world_link:
                    try:
                        world_resp = requests.get(world_link)
                        if world_resp.status_code == 200:
                            world_info = world_resp.json()
                            world_name = world_info.get('name')
                            if world_name and world_name not in planets:
                                planets.append(world_name)
                    except requests.RequestException:
                        continue
                else:
                    if 'unknown' not in planets:
                        planets.append('unknown')

        # See if more pages
        if not data.get('next'):
            break
        page += 1

    return planets
