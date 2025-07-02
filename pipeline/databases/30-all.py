#!/usr/bin/env python3
"""Module that has a function to list all docs in a MongoDB collection"""


def list_all(mongo_collection):
    """Function that lists all documents in a collection:
        mongo_collection: pymongo collection object
        Return: all docs in collection or empty list if no doc in the
        collection"""
    documents = []
    for document in mongo_collection.find():
        documents.append(document)
    return documents