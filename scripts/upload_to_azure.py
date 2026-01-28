from azure.storage.blob import BlobServiceClient
import os

# ⚠️ FILL IN YOUR VALUES
STORAGE_ACCOUNT_NAME = "instacartdatalake"
STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_KEY")  # ✅ Use environment variableCONTAINER_NAME = "bronze-data"

# Path to your downloaded Instacart files
LOCAL_DATA_PATH = "data/"  # Change this to wherever you unzipped the files

def upload_file(blob_service_client, local_file_path, blob_name):
    """Upload a single file to Azure Blob Storage"""
    
    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME,
        blob=blob_name
    )
    
    file_size = os.path.getsize(local_file_path) / (1024 * 1024)  # MB
    print(f"Uploading {blob_name} ({file_size:.2f} MB)...")
    
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    print(f"  ✅ Uploaded!")

def main():
    print("=" * 70)
    print("UPLOADING INSTACART DATA TO AZURE ADLS")
    print("=" * 70)
    
    # Create connection string
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={STORAGE_ACCOUNT_NAME};"
        f"AccountKey={STORAGE_ACCOUNT_KEY};"
        f"EndpointSuffix=core.windows.net"
    )
    
    # Create BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Files to upload
    files = [
        "aisles.csv",
        "departments.csv",
        "orders.csv",
        "order_products__prior.csv",
        "order_products__train.csv",
        "products.csv"
    ]
    
    print(f"\nUploading to:")
    print(f"  Storage: {STORAGE_ACCOUNT_NAME}")
    print(f"  Container: {CONTAINER_NAME}")
    print(f"  Folder: raw/")
    print()
    
    # Upload each file
    for filename in files:
        local_path = os.path.join(LOCAL_DATA_PATH, filename)
        blob_name = f"raw/{filename}"  # Upload to 'raw' folder
        
        if os.path.exists(local_path):
            upload_file(blob_service_client, local_path, blob_name)
        else:
            print(f"  ⚠️  File not found: {local_path}")
    
    print("\n" + "=" * 70)
    print("✅ ALL FILES UPLOADED!")
    print("=" * 70)

if __name__ == "__main__":
    main()