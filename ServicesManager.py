
import time
import configparser

import requests

class ServicesManager(object):
    def __init__(self):
        super(ServicesManager, self).__init__()

        # Set initial call values
        self._last_request_name = None
        self._last_request_payload = None
        self._last_request_time = 0

        # Config parser
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')

        self._api_handle = cfg['SERVICES']['APIHandle']

        self._call_delay = float(cfg['SERVICES']['CallWaitInterval'])
        self._bypass_repeated_calls = cfg['SERVICES']['ServicesBypassRepetedCalls'].split(',')

        self._available_tags = cfg['SERVICES']['AvailableTags'].split(',')
        if len(self._available_tags) == 0:
            raise ValueError('No device tags available to interact with. Check config file.')

        # Retrieve available services
        self._available_services = requests.get(self._api_handle).json()
        self._available_services = [s['name'] for s in self._available_services]

    def call(self, name, targets, payload={}):

        # Check for permission to call services and device tags
        if name not in self._available_services:
            raise NameError('Call to unavailable or unknown service.')
        if targets is None or len(targets) == 0 or targets not in self._available_tags:
            raise NameError('Call to unavailable or unknown device tag.')

        payload['target_tags'] = targets

        # Check for call repetitions
        if name in self._bypass_repeated_calls or \
            name != self._last_request_name or \
            (name == self._last_request_name and payload != self._last_request_payload):

            # Check for time delay between calls
            tnow = time.time()
            if tnow - self._last_request_time > self._call_delay:
                self._last_request_name = name
                self._last_request_payload = payload
                self._last_request_time = tnow

                return self._call(name, payload)

        return False

    def _call(self, name, payload):
        try:
            req = requests.put(self._api_handle + '/' + name,
                               headers={'content-type':'application/json'},
                               json=payload)
        except requests.exceptions.RequestException as e:
            print(e)
            return False
        return True