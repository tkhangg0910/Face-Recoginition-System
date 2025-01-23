from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility

connections.connect("default", host="localhost", port="19530")

collection_name = "user_collection"

def initialize_collection():
    if collection_name not in utility.list_collections():
        print(f"Create collection: {collection_name}")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, max_length=128, auto_id=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        ]
        schema = CollectionSchema(fields, "user collection")
        collection = Collection(collection_name, schema,)
        
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
    else:
        print(f"reference to collection: {collection_name}")
        collection = Collection(collection_name)
    
    collection.load()
    return collection

def get_collection():
    return initialize_collection()
