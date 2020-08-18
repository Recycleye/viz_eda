import os

from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

# connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
connect_str = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey="
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
    "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
)
# Create the BlobServiceClient object which will be used to get a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = "test"
container_client = blob_service_client.get_container_client(container_name)


# def download_blobs(max_concurrency=1):
# List the blobs in the container
print("\nDownloading blobs...")
blob_list = container_client.list_blobs()
os.chdir("./blob_data")
for blob in tqdm(blob_list):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob.name
    )
    if "/" in "{}".format(blob.name):
        head, tail = os.path.split("{}".format(blob.name))
        download_file_path = os.path.join(os.getcwd(), head, tail)
        download_file_folder = os.path.join(os.getcwd(), head)
        if not os.path.isdir(download_file_folder):
            print("Directory doesn't exist, creating it now")
            print(download_file_folder)
            os.makedirs(download_file_folder, exist_ok=True)
    else:
        download_file_path = blob.name
    with open(download_file_path, "wb") as download_file:
        file = blob_client.download_blob(max_concurrency=1)
        download_file.write(file.readall())
