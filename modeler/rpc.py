# Much of this code was built by Agile Scientific. The original can be found here:
# https://github.com/agile-geoscience/notebooks/blob/master/Query_the_RPC.ipynb
# It's licensed under the CC Attribution 4.0

import requests
import pandas as pd

class RPC(object):
    def __init__(self):
        pass
    
    def _query_ssw(self, filters, properties, options):
        
        base_url = "http://www.subsurfwiki.org/api.php"
        q = "action=ask&query=[[RPC:%2B]]"
        q += ''.join(filters) if filters else ''
        q += '|%3F' + '|%3F'.join(properties) if properties else ''
        q += '|' + '|'.join(options) if options else ''
        q += '&format=json'
        
        return requests.get(base_url, params=q)
    
    def _get_formats(self, response):
        formats = {}
        for item in response.json()['query']['printrequests']:
            if item[u'mode'] == 1:
                formats[item[u'label']] = item[u'typeid'].lstrip('_')
        return formats
    
    def _build_dataframe(self, response):
        """
        Takes the response of a query and returns a pandas
        dataframe containing the results.
        """
        try:
            s = list(response.json()['query']['results'].keys())
        except Exception as e:
            raise e
        samples = [i[4:] for i in s]
        df = pd.DataFrame(samples)

        # We'll need to know the formats of the columns.
        formats = self._get_formats(response)
        properties = formats.keys()

        # Now traverse the JSON and build the DataFrame.
        for prop in properties:
            temp = []
            for row in list(s):
                p = response.json()['query']['results'][row]['printouts']
                if p[prop]:
                    if formats[prop] == 'qty':   # Quantity, number + unit
                        temp.append(p[prop][0]['value'])
                    elif formats[prop] == 'wpg':  # Wiki page
                        temp.append(p[prop][0]['fulltext'])
                    else:                         # Anything else: num, txt, tem, etc.
                        temp.append(p[prop][0])
                else:
                    temp.append(None)
            df[prop] = temp
        
        df = df.set_index(0)
        df.index.name = None

        return df
    
    def query(self, filters=None, properties=None, options=None):
        r = self._query_ssw(filters, properties, options)
        if r.status_code == 200:
            return self._build_dataframe(r)
        else:
            print("Something went wrong.")
        

rpc = RPC()

filters = ["[[lithology::Shale||Sandstone||Limestone]][[Delta::%2B]]"]
properties = ['Citation', 'Description', 'Lithology', 'Vp', 'Vs', 'Rho', 'Delta', 'Epsilon']
options = ["limit=100"]

df = rpc.query(filters, properties, options)
df.head()
