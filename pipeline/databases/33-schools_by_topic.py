#!/usr/bin/env python3
"""Module that provides a function to find schools by topic"""


def schools_by_topic(mongo_collection, topic):
    """Function that returns the list of school having a specific topic:
        mongo_collection: the pymongo collection object
        topic: (string) will be topic searched"""
    schools = []
    for school in mongo_collection.find({"topics": topic}):
        schools.append(school)
    return schools