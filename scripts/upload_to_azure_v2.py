from azure.storage.blob import BlobServiceClient
import os
import time

# ⚠️ FILL IN YOUR VALUES
STORAGE_ACCOUNT_NAME = "instacartdatalake"
STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_KEY")  # ✅ Use environment variableCONTAINER_NAME = "bronze-data"
CONTAINER_NAME = "bronze-data"
LOCAL_DATA_PATH = "data/"  # Your data folder

def check_if_exists(blob_service_client, blob_name):
    """Check if blob already exists"""
    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME,
        blob=blob_name
    )
    return blob_client.exists()

def upload_file_with_retry(blob_service_client, local_file_path, blob_name, max_retries=3):
    """Upload with retry logic and better timeout handling"""
    
    file_size = os.path.getsize(local_file_path) / (1024 * 1024)  # MB
    
    # Check if already uploaded
    if check_if_exists(blob_service_client, blob_name):
        print(f"  ⏭️  Already exists, skipping...")
        return True
    
    print(f"Uploading {blob_name} ({file_size:.2f} MB)...")
    
    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME,
        blob=blob_name
    )
    
    for attempt in range(max_retries):
        try:
            with open(local_file_path, "rb") as data:
                # Use longer timeout for large files
                blob_client.upload_blob(
                    data, 
                    overwrite=True,
                    timeout=600  # 10 minutes timeout
                )
            
            print(f"  ✅ Uploaded!")
            return True
            
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt + 1} failed: {str(e)[:100]}...")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                print(f"  ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Failed after {max_retries} attempts")
                return False

def main():
    print("=" * 70)
    print("RESUMING UPLOAD TO AZURE ADLS")
    print("=" * 70)
    
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={STORAGE_ACCOUNT_NAME};"
        f"AccountKey={STORAGE_ACCOUNT_KEY};"
        f"EndpointSuffix=core.windows.net"
    )
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    files = [
        "aisles.csv",
        "departments.csv",
        "orders.csv",
        "order_products__prior.csv",
        "order_products__train.csv",
        "products.csv"
    ]
    
    print(f"\nChecking and uploading files to:")
    print(f"  Storage: {STORAGE_ACCOUNT_NAME}")
    print(f"  Container: {CONTAINER_NAME}")
    print(f"  Folder: raw/")
    print()
    
    success_count = 0
    failed_files = []
    
    for filename in files:
        local_path = os.path.join(LOCAL_DATA_PATH, filename)
        blob_name = f"raw/{filename}"
        
        if os.path.exists(local_path):
            if upload_file_with_retry(blob_service_client, local_path, blob_name):
                success_count += 1
            else:
                failed_files.append(filename)
        else:
            print(f"  ⚠️  File not found: {local_path}")
            failed_files.append(filename)
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully uploaded: {success_count}/{len(files)} files")
    
    if failed_files:
        print(f"❌ Failed files: {', '.join(failed_files)}")
    else:
        print("🎉 ALL FILES UPLOADED!")
    
    print("=" * 70)

if __name__ == "__main__":
    main()