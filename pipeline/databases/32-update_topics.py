#!/usr/bin/env python3
"""Module for function"""


def update_topics(mongo_collection, name, topics):
    """Function that changes all topics of a school doc based on name
    name: (string) will be the school name to update
    topics: (list of strings) will be the list of topics approached in the school"""
    mongo_collection.update_many(
        {"name": name},
         {"$set": {"topics": topics}}
    )
