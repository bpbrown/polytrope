"""store_metadata.py

Extract metadata about a run from its p0 log file.

Usage:
    store_metadata.py <base_path> [--output=<output> --unjoined]

Options:
    --output=<output>  Output directory; if blank a guess based on likely case name will be made
"""
import csv
import re 
import os
from docopt import docopt

import argparse
import httplib2
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

# replace this
flags = argparse.Namespace(noauth_local_webserver=True,logging_level='ERROR')
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Google Sheets API Python Quickstart'
def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'sheets.googleapis.com-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

class Metadata():
    def __init__(self, base_path, cfg_file='metadata.csv'):
        self.base_path = base_path
        self.logfilename = os.path.join(base_path,'logs','dedalus_log_p0.log')
        self.re = {}
        self.dtype = {}
        self.values = {}
        self.cfg_file = cfg_file
        self.read_init()

    def read_init(self):
        with open(self.cfg_file,"r") as csvfile:
            reader = csv.reader(csvfile)
            for r in reader:
                k,re,dtype = r
                self.re[k] = re
                self.dtype[k] = self.dtype_func(dtype)

    def dtype_func(self,dtype_str):
        if dtype_str == 'int':
            dtype = int
        elif dtype_str == 'float':
            dtype = float
        elif dtype_str == '2*int':
            dtype = lambda x: 2*int(x)
        else:
            dtype = str
        return dtype

    def get_metadata(self):
        # always add run name
        bp = self.base_path.rstrip('/')
        self.values['Name'] = bp.split('/')[-1]
        
        for k,regexp in self.re.items():
            self.values[k] = self.dtype[k](self.find(regexp))

    def find(self, regexp):
        with open(self.logfilename,"r") as log:
            for l in log:
                m = re.search(regexp,l)
                if m:
                    break
        return m.groups(1)[0]

    def upload_metadata(self,sheetName,spreadsheetID='1guRQbfBF3hPfjZhNSiO88ni_fR95xOIPpA6JelhkgeU'):
        credentials = get_credentials()
        http = credentials.authorize(httplib2.Http())
        discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                        'version=v4')
        service = discovery.build('sheets', 'v4', http=http,
                                  discoveryServiceUrl=discoveryUrl)
        
        headerRange= '{}!1:1'.format(sheetName)
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheetID, range=headerRange).execute()
        values = result.get('values',[])
        if not values:
            raise ValueError("No headers in the {} sheet. Make sure the sheet has a first row with the names of metadata you want to store.".format(sheetName))
        else:
            headers = values[0]

        send_values = []
        for h in headers:
            try:
                send_values.append(self.values[h])
            except KeyError:
                print("Metadata key {} not found. Skipping...".format(h))
        
        sendRange = '{}!1:1'.format(sheetName)
        values = [send_values,]
        value_input_option = "USER_ENTERED"
        body = {'values':values}
        result = service.spreadsheets().values().append(spreadsheetId=spreadsheetID, range=sendRange, valueInputOption=value_input_option,body=body).execute()

if __name__ == "__main__":
    args = docopt(__doc__)

    base_path = args['<base_path>']
    metadata = Metadata(base_path)
    metadata.get_metadata()
    metadata.upload_metadata("Equilibration Study")
