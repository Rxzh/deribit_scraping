import json
import datetime
import time
import numpy as np
from sklearn.metrics import mean_squared_error

def somme_ponderee(X):
    """
    Input: str représentant une liste de listes de floats = '[[a1,b1],[a2,b2],...,[an,bn]]'
    Output: (Somme des ak*bk) / (Somme des bk)
    """
    s = 0
    d = 0
    
    if type(X) != list:
        X = json.loads(X)
    for x in X:

        if len(x)<2:
            return 0.0
        s += x[0]*x[1]
        d += x[1]
    if d!= 0:
        s/= d
    else:
        return 0.0
    return s

def string_to_dict(s):
    """
    Input: str représentant un dictionnaire
    Output: Le dit dictionnaire
    """
    json_acceptable_string = s.replace("'", "\"").replace("None","0.0")
    return json.loads(json_acceptable_string)

def to_date(ts):
    """
    Input: timestamp = nombre de secondes depuis le 01/01/1970
    Output: date
    """
    return str(datetime.datetime.fromtimestamp(ts))


def to_ts(date):
    """
    Input: Date, au format {dd/mm/yy, yy-mm-dd, 13DEC21}
    Output: timestamp en secondes
    """

    
    if '/' in date: #Format dd/mm/yy
        return int(time.mktime(datetime.datetime.strptime(date, "%d/%m/%Y").timetuple()))
    elif '-' in date: #Format yy-mm-dd
        return int(time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple()))
    else: # FORMAT 13DEC21
        if len(date)<7:
            date = '0'+date

            
        day = date[:2]
        year = date[-2:]
        month = date[2:-2]

        month_d = { 'JAN':'01',
                    'FEB':'02',
                    'MAR':'03',
                    'APR':'04',
                    'MAY':'05',
                    'JUN':'06',
                    'JUL':'07',
                    'AUG':'08',
                    'SEP':'09',
                    'OCT':'10',
                    'NOV':'11',
                    'DEC':'12'}


        return to_ts(day+'/'+month_d[month]+'/20'+year)
    
def ms_to_s(ts):
    """
    Input: timestamp en ms
    Outpu: timestamp en s
    """
    return int(ts/1000)


def ts_to_years(ts):
    """
    Input: timestamp en s
    Output: nombre de jours, float
    """
    return ts/(3600*24*365)



def pipeline(df, option_type = 'B'):
    
    if option_type != 'B':
        df = df[df.option_type == option_type]


    df['mid_iv'] = (df.bid_iv + df.ask_iv)/2

    ### ajouts des volumes


    if type(list(df['stats'])[0]) == dict:
        VOLUMES = [x['volume'] for x in df['stats']]
    else :
        VOLUMES = [x['volume'] for x in df['stats'].apply(string_to_dict)]
        
    df.insert(4, 'V', VOLUMES)
        
    #### Ajout des Strikes K et des dates de maturité _T 

    EXP_DATES, STRIKES = [], []
    for inst_name in df['instrument_name']:
        el = inst_name.split('-')
        EXP_DATES.append(el[1])
        STRIKES.append(el[2])
    
    df.insert(4, '_T', EXP_DATES)
    df.insert(4, 'K', STRIKES)
    
    #### Changement de _T pour le nombre de jours restants avant l'expiration du contrat (à partir du call de la fonction)
    df['_T'] = df['_T'].apply(to_ts) - datetime.datetime.timestamp(datetime.datetime.today())#btc_subset['timestamp'].apply(ms_to_s)
    df['_T'] = df['_T'].apply(ts_to_years)

    
    #Equivalent a : 
    #df['K'] = STRIKES
    #df['_T'] = EXP_DATES
    
    
    #### Ajout des bids/asks/mid
    df['bids'] = df['bids'].apply(somme_ponderee)
    df['asks'] = df['asks'].apply(somme_ponderee)
    df['mid'] = (df.bids+df.asks)/2
    

    #### Supression des lignes ou les IV sont nulles et où les Volumes sont nuls
    #df = df.drop(df[df.mid_iv == 0.0].index)
    df = df.drop(df[df.V <= 5.0].index)
    df = df.drop(df[df.K.astype(float) >= 80000.0].index)
    df = df.drop(df[df._T.astype(float) >= 0.25].index)
    df = df.drop(df[df._T.astype(float) <= 0.0].index)

    #### Supression des lignes ou les IV sont nulles et où les Volumes sont nuls
    #moy_volumes = sum(df.V)/len(df.V)
    #var_volumes = np.sqrt(sum((np.array(df.V)-moy_volumes)**2)/len(df.V))
    #min_acceptable_volume= moy_volumes-0.3*var_volumes
    
    #print("moy_volumes = ", moy_volumes)
    #print("var_volumes = ", var_volumes)
    #print("min_acceptable_volume = ", min_acceptable_volume)
    
    #df = df.drop(df[df.V <= min_acceptable_volume].index)
    

    
    #### Ajout du prix du sous-jacent 
    df['S'] = df['underlying_price']
    
    #### Tri selon les Strikes croissants
    #df = df.sort_values('K')

    #### Tri selon les T croissants
    #df = df.sort_values('_T')
    #df = df[df._T == list(df['_T'])[0]]

    #### Transformation des IV de % à valeurs dans [0,1]
    df['mid_iv'] = df['mid_iv']#/100 ??
    
    """
    df['IV'] = np.vectorize(IV)(C = df['last_price'].astype(float),
                                S = df['S'].astype(float),
                                K = df['K'].astype(float),
                                T = df['_T'].astype(float))
    """
    #calc_impl_vol(price = 5., right = 'C', underlying = 100., strike = 100., time = 1., rf = 0.0, inc = 0.0001)


    
    
    #### 
    df['moneyness'] = df['S']/df['K'].astype(float)

    final_df = df[['S','K','V', '_T','bids','asks','last_price','mid_iv','mid', 'moneyness']]
    
    final_df = final_df.astype(float)
    
    final_df.insert(8,'option_type', df['option_type'])
    final_df = final_df.dropna()
    final_df = final_df.sort_values(['_T','K'])

    
    return final_df



def MSE(col1,col2):
    if np.size(col1) != np.size(col2) or np.size(col1)==0:
        raise ValueError("col1 and col2 must be the same size and not null")
    return np.sum((col1-col2)**2)/np.size(col1)

def RMSE(col1,col2):
    if np.size(col1) != np.size(col2) or np.size(col1)==0:
        raise ValueError("col1 and col2 must be the same size and not null")
    return mean_squared_error(col1, col2, squared=False)