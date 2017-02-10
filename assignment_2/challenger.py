import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime

import girder_client

GIRDER_URL = 'http://ismi17.diagnijmegen.nl/girder/api/v1'

def submit_results(user, results, description = None):
	client = girder_client.GirderClient(apiUrl=GIRDER_URL)
	upload_challenge_data(client, user, 'drive', results, metadata=description)
	
	
def upload_challenge_data(client, user, challenge_name, results_folder, metadata=None):
    working_directory = tempfile.mkdtemp()
    try:
        results_zip = zip_directory(results_folder, working_directory)
        upload_file_to_server(client, user, results_zip, challenge_name, metadata)
    finally:
        shutil.rmtree(working_directory)


def upload_file_to_server(client, user, file, challenge_name, metadata=None):
    client.authenticate(username=user['username'], password=user['password'])

    # The following will just return the first item in the list!!!
    collection = client.get('collection', {'text': challenge_name, 'limit': 1})[0]
    folder = client.get('folder', {'parentId': collection['_id'],
                                   'parentType': 'collection',
                                   'name': user['username'].lower()})[0]

    item = client.createItem(folder['_id'], 'Submission %s'%datetime.utcnow())

    if metadata is not None:
        client.addMetadataToItem(item['_id'], metadata)

    # Upload file data
    client.uploadFileToItem(item['_id'], file)


def zip_directory(input_dir, output_dir):
    """
    Creates a temporary directory and creates a zipfile with the algorithm result
    :param folder:
    :return:
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    foldername = os.path.split(input_dir)[1]
    temp_zip_file = os.path.join(output_dir, foldername + '.zip')

    with zipfile.ZipFile(temp_zip_file, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), input_dir))

    return temp_zip_file