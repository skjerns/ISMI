"""
This script tests if all the necessary modules are available for the course.
"""

import os
import sys
import cStringIO

if __name__ == "__main__":

    # Redirect stderr to string. Lasagne writes there to report GPU usage.
    #
    stderr_str = cStringIO.StringIO()
    sys.stderr = stderr_str

    # Determine the OS. We install the packages in user mode on Linux.
    #
    user_mode = '' if os.name == 'nt' else '--user '
    
    # Modules to test: module -> package.
    #
    module_list = [('SimpleITK', 'SimpleITK'),
                   ('dicom', 'pydicom'),
                   ('sklearn', 'sklearn'),
                   ('girder_client', 'girder-client'),
                   ('tqdm', 'tqdm'),
                   ('theano', '--upgrade https://github.com/Theano/Theano/archive/master.zip'),
                   ('lasagne', '--upgrade https://github.com/Lasagne/Lasagne/archive/master.zip')]
    
    # First try the importlib.
    #
    print 'Looking for modules:'
    try:
        sys.stdout.write('{module} ... '.format(module='importlib'))
        import importlib
    except:
        print 'missing, try: "pip install {mode}importlib"'.format(mode=user_mode)
        exit()
    else:
        print 'ok'
    
    # Test all modules.
    #
    for module_item in module_list:
        try:
            sys.stdout.write('{module} ... '.format(module=module_item[0]))
            importlib.import_module(module_item[0])
        except:
            print 'missing, try: "pip install {mode}{package}"'.format(mode=user_mode, package=module_item[1])
        else:
            print 'ok'

    # Print messages, if any.
    #    
    stderr_messages = stderr_str.getvalue()    
    if stderr_messages:
        print ''
        print 'Messages:'
        sys.stdout.write(stderr_messages)
