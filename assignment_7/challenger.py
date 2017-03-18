import json
import os
import shutil
import tempfile
import zipfile
import time
import uuid
from datetime import datetime

import girder_client

GIRDER_URL = 'http://ismi17.diagnijmegen.nl/girder/api/v1'

CASE_UID_MAP = {
    'drive': {
        'case_identifier': 'alg_vessel_segmentation_filename',
        'cases': {
            1: '1_mask.png',
            2: '2_mask.png',
            3: '3_mask.png',
            4: '4_mask.png',
            5: '5_mask.png',
            6: '6_mask.png',
            7: '7_mask.png',
            8: '8_mask.png',
            9: '9_mask.png',
            10: '10_mask.png',
            11: '11_mask.png',
            12: '12_mask.png',
            13: '13_mask.png',
            14: '14_mask.png',
            15: '15_mask.png',
            16: '16_mask.png',
            17: '17_mask.png',
            18: '18_mask.png',
            19: '19_mask.png',
            20: '20_mask.png',
      }
    }
}


def submit_results(user, results, description=None):
    client = girder_client.GirderClient(apiUrl=GIRDER_URL)
    upload_challenge_data(client, user, 'drive', results, metadata=description)
    print(
        "You successfully submitted your segmentations. The results of your submission will appear shortly on the leaderboard at http://ismi17.diagnijmegen.nl/.")


def upload_challenge_data(client, user, challenge_name, results_folder, metadata=None):
    working_directory = tempfile.mkdtemp()
    try:
        test_for_all_files(challenge_name, results_folder)
        create_results_csv(challenge_name, results_folder)
        create_challengr_json(challenge_name, results_folder)
        results_zip = zip_directory(results_folder, working_directory)
        upload_file_to_server(client, user, results_zip, challenge_name, metadata)
    finally:
        shutil.rmtree(working_directory)


def upload_file_to_server(client, user, file, challenge_name, metadata=None):
    print('Uploading results to server')
    client.authenticate(username=user['username'], password=user['password'])

    # The following will just return the first item in the list!!!
    collection = client.get('collection', {'text': challenge_name, 'limit': 1})[0]
    folder = client.get('folder', {'parentId': collection['_id'],
                                   'parentType': 'collection',
                                   'name': user['username'].lower()})[0]

    item = client.createItem(folder['_id'], 'Submission %s' % datetime.utcnow())

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
    print('Compressing results')
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    foldername = os.path.split(input_dir)[1]
    temp_zip_file = os.path.join(output_dir, foldername + '.zip')

    with zipfile.ZipFile(temp_zip_file, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), input_dir))

    return temp_zip_file

def test_for_file(filename, folder_listing):
    if not filename in folder_listing:
        raise(ValueError(filename + ' not found in results folder. Have you run the algorithm on all of the test data?'))


def test_for_all_files(challenge_name, results_folder):
    print('Testing that all the results are in %s'%results_folder)
    results_files = os.listdir(results_folder)
    for caseId in CASE_UID_MAP[challenge_name]['cases']:
        test_for_file(CASE_UID_MAP[challenge_name]['cases'][caseId],
                      results_files)

def create_results_csv(challenge_name, results_folder):
    filename = os.path.join(results_folder, 'algorithm_result.csv')
    print('Creating %s'%filename)

    with open(filename, 'w') as f:
        f.write('caseid,%s\n'%CASE_UID_MAP[challenge_name]['case_identifier'])
        for caseId in CASE_UID_MAP[challenge_name]['cases']:
            f.write('\"%s\",%s\n'%(caseId, CASE_UID_MAP[challenge_name]['cases'][caseId]))


def create_challengr_json(challenge_name, results_folder):
    json_filename = os.path.join(results_folder, 'challengr.json')
    print('Creating %s' % json_filename)

    foldername = os.path.basename(os.path.normpath(results_folder))

    challengr_metadata = {"timestamp": time.time(), "algorithmfields": {
        "fields": [],
        "description": "",
        "uuid": str(uuid.uuid1()),
        "name": ""}, "uid": "", "timings": {}, "parametrization": {}, 'uid': str(foldername)}

    with open(json_filename, 'w') as f:
        f.write(json.dumps(challengr_metadata))
