from db_config import get_collection
import numpy as np
def insert_user(data, user_collection):
    try:
        user_collection.insert(data)
        return True
    except Exception as e:
        print(f"Error inserting data: {e}")
        return False

def search_user(embedding, user_collection):
    user_collection.load()
    results = user_collection.search([embedding], "embedding", param={"metric_type": "L2", "params": {"nprobe": 10}}, output_fields=["name"], limit=1)
    if results and len(results[0]) > 0:  
        first_result = results[0][0]    
        print(first_result)
        if first_result.distance < 1:  
            return first_result.name


def show_data(user_collection):
    try:
        
        results = user_collection.query(
            expr="",
            output_fields=["name", "embedding"],
            limit=10 
        )
        
        print("Data in the collection:")
        for record in results:
            print(f"User ID: {record['name']}, Vectors: {record['embedding']}")
        
        return results

    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None


