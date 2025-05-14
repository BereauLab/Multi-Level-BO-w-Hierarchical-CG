import numpy as np
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
database = client.get_database("molecules-4")
collection = database.get_collection("level-2")

mols_per_file = 500000
counter = 0
while True:
    print(f"Processing {counter}", end="\r")
    cursor = list(
        collection.find({}, {"latent_space": 1, "name": 1})
        .skip(counter * mols_per_file)
        .limit(mols_per_file)
    )
    latent_space = np.array([doc["latent_space"] for doc in cursor])
    names = [doc["name"] for doc in cursor]
    np.savez(f"latent_space_{counter}.npz", latent_space=latent_space, names=names)
    counter += 1
    if len(latent_space) < mols_per_file:
        break
