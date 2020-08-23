import os
from datetime import datetime, timedelta

from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv(override=True)
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
account_key = os.getenv("AZURE_STORAGE_ACCOUNT_ACCESS_KEY")
# Create the BlobServiceClient object used to get a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# Create the ContainerClient object used to get blob data
container_name = "datastorage"
container_client = blob_service_client.get_container_client(container_name)


def get_blob_datasets():
    folders = container_client.walk_blobs(delimiter="/")
    datasets = [f.name[:-1] for f in folders]
    return datasets


def get_blobs(dataset):
    blob_list = container_client.list_blobs(name_starts_with=dataset)
    return blob_list


def get_blob_url(blob_name):
    token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1),
    )
    url = (
        f"https://{account_name}"
        f".blob.core.windows.net/{container_name}/{blob_name}?{token}"
    )
    return url


def download_blobs(blob_list):
    # List the blobs in the container
    print("\nDownloading blobs...")
    if not os.path.exists("./blob_data"):
        os.makedirs("./blob_data")
    os.chdir("./blob_data")
    blob_names = []
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
        blob_names.append(blob.name)
    return blob_names
