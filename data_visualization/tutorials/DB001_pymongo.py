# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:56:32 2018

@author: zouco
"""

from pprint import pprint

from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
data = {}
db = client.examples # use example database
db.collection1.insert(data)
for a in db.collection1.find():
    pprint(a)