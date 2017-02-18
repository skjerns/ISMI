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
    'exact09': {
        'case_identifier': 'alg_airway_segmentation_uid',
        'cases': {
            '0': '1.0.000.000000.0.00.0.0000000000.0000.0000000000.000',
            '1': '1.2.276.0.28.3.0.14.4.0.20090213134050413',
            '2': '1.2.276.0.28.3.0.14.4.0.20090213134114792',
            '3': '1.2.392.200036.9116.2.2.2.1762676169.1080882991.2256',
            '4': '1.2.840.113704.1.111.2004.1131987870.11',
            '5': '1.2.840.113704.1.111.2296.1199810886.7',
            '6': '1.2.840.113704.1.111.2296.1199810941.11',
            '7': '1.2.840.113704.1.111.4400.1131982359.11',
            '8': '1.3.12.2.1107.5.1.4.50585.4.0.7023259421321855',
            '9': '2.16.840.1.113669.632.21.3825556854.538251028.390606191418956020'},
        'dimensions': {
            '0': '512 512 476',
            '1': '512 512 381',
            '2': '512 512 376',
            '3': '512 512 501',
            '4': '512 512 297',
            '5': '512 512 498',
            '6': '512 512 455',
            '7': '512 512 318',
            '8': '512 512 391',
            '9': '512 512 267',
        }
    }
}


def submit_results(user, results, description=None):
    client = girder_client.GirderClient(apiUrl=GIRDER_URL)
    upload_challenge_data(client, user, 'exact09', results, metadata=description)
    print(
        "You successfully submitted your segmentations. The results of your submission will appear shortly on the leaderboard at http://ismi17.diagnijmegen.nl/.")


def upload_challenge_data(client, user, challenge_name, results_folder, metadata=None):
    working_directory = tempfile.mkdtemp()
    try:
        test_for_all_files(challenge_name, results_folder)
        test_for_mhd_dimensions(challenge_name, results_folder)
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

def test_for_file(filename, extension, folder_listing):
    if not (filename + '.' + extension in folder_listing or
        filename + '.' + extension.upper() in folder_listing or
        filename + '.' + extension.lower() in folder_listing):
        raise(ValueError(filename + '.' + extension + ' not found in results folder. Have you run the algorithm on all of the test data?'))

def test_for_mhd_dimensions(challenge_name, results_folder):
    print('Checking MHD Dimensions')
    for caseId in CASE_UID_MAP[challenge_name]['cases']:

        mhd_file = os.path.join(results_folder,CASE_UID_MAP[challenge_name]['cases'][caseId] + '.mhd')

        with open(mhd_file, 'r') as f:
            mhd_content = f.readlines()

        mhd_content = ' '.join(mhd_content)
        dimsize_str = 'DimSize = ' + CASE_UID_MAP[challenge_name]['dimensions'][caseId]

        if dimsize_str not in mhd_content:
            raise(ValueError('The dimensions of %s should be %s, please check your algorithm and .mhd file'%
                             (CASE_UID_MAP[challenge_name]['cases'][caseId],
                              CASE_UID_MAP[challenge_name]['dimensions'][caseId])))


def test_for_all_files(challenge_name, results_folder):
    print('Testing that all the results are in %s'%results_folder)
    results_files = os.listdir(results_folder)
    for caseId in CASE_UID_MAP[challenge_name]['cases']:
        test_for_file(CASE_UID_MAP[challenge_name]['cases'][caseId],
                      'mhd',
                      results_files)
        test_for_file(CASE_UID_MAP[challenge_name]['cases'][caseId],
                      'zraw',
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
