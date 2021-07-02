import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

class Scraper:
    def __init__(self, currency="BTC"):
        """
        Initialise la classe `Scraper`.

        Paramètres
        ----------
        currency: string ("BTC" ou "ETH")
            
            La crypto-monnaie qui sera utilisée pour récupérer toutes les données relatives aux options et à la volatilité.
            Le paramètre par défaut est "BTC".

        Exemple
        ---------
        >>> import deribit_data as dm
        >>> data = dm.Scraper(currency = "BTC")
        """

        self.url = 'https://www.deribit.com/api/v2/public/'
        self.currency = str.lower(currency)

    def get_hist_vol(self, save_csv=False):
        """
        Récupère la volatilité historique annualisée de l'actif, mesurée toutes les heures au cours des 15 derniers jours.
        
        Paramètres
        ======================
        Input: save_csv: boolean, pour choisir ou non de sauvegarder en csv.
        =======================
        Output: dataframe des volatilités

        Exemple
        -------------
        >>> df = data.get_hist_vol()
        >>> df.tail()
        date                    btc_hist_vol
        2020-05-13 21:00:00     95.228999
        2020-05-13 22:00:00     95.501434
        2020-05-13 23:00:00     95.553333
        2020-05-14 00:00:00     95.572422
        2020-05-14 01:00:00     95.564853
        """
        data = {"currency": self.currency}
        d = requests.get(self.url + "get_historical_volatility?", data)
        df = pd.DataFrame(d.json()['result'])
        df.columns = ['date', str(self.currency.lower())+'_hist_vol']
        df['date'] = pd.to_datetime(df.date, unit='ms')
        df = df.set_index("date")
        if save_csv :
            df.to_csv(str(self.currency) + "_hist_vol.csv")
        return df

    def get_options_list(self):
        """
        Cette méthode est utilisée pour récupérer le type d'option et le nom de l'instrument.
        La sortie de cette méthode est utilisée comme entrée dans la méthode `get_option_stats'.
        de la méthode "get_option_stats".
        =======================
        Output  dataframe

        Exemple
        -------------
        >>> df = data.get_options_list()
        >>> df.head()
        expiration_timestamp option_type instrument_name strike
        0	1590134400000	put	    BTC-22MAY20-9250-P  9250.0
        1	1590134400000	call	BTC-22MAY20-12000-C	12000.0
        2	1593158400000	call	BTC-26JUN20-8000-C	8000.0
        3	1589443200000	call	BTC-14MAY20-8875-C	8875.0
        4	1601020800000	put	    BTC-25SEP20-9000-P  9000.0
        """
        data = {'currency': self.currency, 'kind': 'option'}
        r = requests.get(self.url + 'get_instruments', data)
        df = pd.DataFrame(r.json()['result'])
        cols = ['expiration_timestamp', 'option_type', 'instrument_name', 'strike']
        return df[cols]

    def get_option_urls(self):
        """
        Utilisé pour récupérer les liens URL pour toutes les options qui sont saisies dans la méthode
        méthode `collect_data`.

        Returns
        -------------
        list
            

        Exemple
        -------------
        >>> data.get_options_urls()
        ['https://www.deribit.com/api/v2/public/get_order_book?instrument_name=BTC-4SEP20-13250-P',
        'https://www.deribit.com/api/v2/public/get_order_book?instrument_name=BTC-26MAR21-8000-P',
        ....]
        """
        url_storage = []
        options_list = self.get_options_list()
        request_url = self.url + 'get_order_book?instrument_name='
        for option in range(len(options_list)):
            data = request_url + options_list.instrument_name.values[option]
            url_storage.append(data)
        return url_storage


    def request_get(self, url):
        """
        Une fonction intermédiaire utilisée en conjonction avec la méthode `collect_data`.
        """
        page = requests.get(url)
        time.sleep(1)
        return page.json()['result']


    def collect_data(self, max_workers=20, save_csv=False):
        """
        Récupère le prix, la volatilité implicite, le volume, 
        les greeks et d'autres données pertinentes pour toutes les options de l'actif sélectionné.

        ========================
        Input:  max_workers: integer
                Sélectionne le nombre maximum de threads pour exécuter les appels de manière asynchrone (par défaut 20).
                -----------------
                save_csv: boolean
                Sélectionnez True pour sauvegarder les données des options dans un fichier csv. (False par défaut.)
        =========================
        Output: dataframe:
                un dataframe avec les connées correspondantes pour chaque option.
                -----------------
                csv:
                Un fichier csv avec ces données, si save_csv est à True
        =========================


        Exemple
        -------------
        >>> df = data.collect_data()
        >>> df.columns
        Index(['expiration_timestamp', 'option_type', 'instrument_name', 'strike',
               'underlying_price', 'underlying_index', 'timestamp', 'stats', 'state',
               'settlement_price', 'open_interest', 'min_price', 'max_price',
               'mark_price', 'mark_iv', 'last_price', 'interest_rate',
               'instrument_name', 'index_price', 'greeks', 'estimated_delivery_price',
               'change_id', 'bids', 'bid_iv', 'best_bid_price', 'best_bid_amount',
               'best_ask_price', 'best_ask_amount', 'asks', 'ask_iv'],
                dtype='object')

        >>> df[['instrument_name', 'last_price', 'mark_iv', 'open_interest']].head()
            instrument_name	  last_price mark_iv open_interest
        0	BTC-22MAY20-9250-P	0.0460	77.72	138.8
        1	BTC-22MAY20-12000-C	0.0050	112.28	110.6
        2	BTC-26JUN20-8000-C	0.1825	90.26	771.0
        3	BTC-14MAY20-8875-C	0.0410	90.65	24.7
        4	BTC-25SEP20-9000-P	0.1770	83.37	266.9
        """
        raw_data = []
        pool = ThreadPoolExecutor(max_workers=max_workers)
        print("Collecting data...")
        for asset in pool.map(self.request_get, self.get_option_urls()):
            raw_data.append(asset)
        df = pd.DataFrame(raw_data)
        df['option_type'] = [df.instrument_name[i][-1] for i in range(len(df))]
        df = df.loc[:, ~df.columns.duplicated()]
        if save_csv:
            label = datetime.now().strftime(str(self.currency) + "_options_data" +'-%Y_%b_%d-%H_%M_%S.csv')
            df.to_csv(label)
        print("Data Collected")
        return df
