#!/usr/bin/env python3
"""Module that has a function to insert a new doc in mongodb collection"""


def insert_school(mongo_collection, **kwargs):
    """Function that inserts a new document in a collection based on kwargs
    Returns: new _id"""
    r = mongo_collection.insert_one(kwargs)
    return r.inserted_id