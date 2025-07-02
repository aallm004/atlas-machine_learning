#!/usr/bin/env python3
"""Script that provides stats about Nginx logs stored in MongoDB"""
import pymongo


def log_stats():
    """Function that provides some stats about Nginx logs stored in MongoDB
        Database: logs
        Collection: nginx"""
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs
    nginx = collection.nginx

    # Get number of logs
    total_logs = collection.count_documents({})
    print(f"{total_logs} logs")

    # Method stats
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")

    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    # Get status check count
    status_check = nginx.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_check} status check")

    if __name__ == "__main__":
        log_stats()