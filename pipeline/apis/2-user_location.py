#!/usr/bin/env python3
"""GitHub API user location script"""
import requests
import sys
from datetime import datetime


def get_user_location(api_url):
    """Script that prints the location of a specific user"""
    try:
        response = requests.get(api_url)

        # Handle rate limit (403)
        if response.status_code == 403:
            reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
            current_time = int(datetime.now().timestamp())
            minutes_left =  max(0, (reset_time - current_time) // 60)
            print(f"Reset in {minutes_left} min")


        # User not found (404)
        if response.status_code == 404:
            print("Not found")
            return
        
        # Handle successful response (200)
        if response.status_code == 200:
            user_data = response.json()
            location = user_data.get('location')

            if location:
                print(location)
            else:
                print ("Not found")
        else:
            print("Not Found")
    except requests.RequestException:
        print("Not found")
