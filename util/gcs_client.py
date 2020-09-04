from google.cloud import storage
import requests

class GCSclient:
    def __init__(self,project,bucket_name,max_retry=5,verb=True):
        self.project = project
        self.bucket_name = bucket_name
        self.max_retry = max_retry
        self.verb = verb
        self.init_client()

    def init_client(self):
        self.client = GCSclient.add_client_retries(
            storage.Client(project=self.project),
            max_retries = self.max_retry)
        self.bucket = self.client.get_bucket(self.bucket_name)

    @staticmethod
    def add_client_retries(client,max_retries=5):
        """ Retry connection to prevent connectionResetError """
        adapter = requests.adapters.HTTPAdapter(max_retries=max_retries)
        client._http.mount("https://", adapter)
        client._http._auth_request.session.mount("https://", adapter)
        return client

    def upload_blob(self, source_file_name, destination_blob_name):
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        if self.verb: 
            print('File {} uploaded to {}.'.format(source_file_name,destination_blob_name))

    def download_blob(self, source_blob_name, destination_file_name):
        blob = self.bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        if self.verb: 
            print('File {} downloaded from {}.'.format(destination_file_name,source_blob_name))

    def rename_blob(self,blob_name, new_name,pre=""):
        bucket = self.bucket
        blob = bucket.blob(pre+blob_name)
        new_blob = bucket.rename_blob(blob, pre+new_name)
        print("Blob {} has been renamed to {}".format(blob.name, new_blob.name))