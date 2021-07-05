```python
#!code .
```

# Pricing d'option sur cryptomonnaies
#### Introduction

### Imports

Les imports se font de la manière suivante
- Le scraper se trouve dans le module Market
- Les modèles sont dans le module Models
- Des fonctions utiles se trouvent dans le sous-module utils du module Models, on y retrouve des fonctions d'évaluations pour les modèles


```python
from Cryptocurrency_pricing.Market import deribit_data as dm
from Cryptocurrency_pricing.Models.utils import *
from Cryptocurrency_pricing.Models import BlackScholes, Merton, Heston

import numpy as np
import plotly.graph_objects as go   
```

### Scraping

Pour effectuer le scraping de données depuis deribit, il faut instancier la classe Scraper du module deribit_data, l'argument $\verb+currency+$ permet de choisir la cryptomonnaie sur laquelle on veut récupérer les données du marché (BTC, ETH, etc...) 


```python
data = dm.Scraper(currency='BTC')
```

Une fois le scraper initialisé, on collecte la donnée par la méthode $\verb+collect_data+$.
- $\tt{max\_workers}$ determine le nombre de threads qui vont marcher en parallèle pour scraper la donnée (20 max pour ne pas surcharger les requêtes)
- $\tt{save\_csv = True}$ pour sauvegarder la donnée scrapée en csv.



```python
raw_df = data.collect_data(max_workers = 15, save_csv = False)
```

    Collecting data...
    Data Collected



```python
raw_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>underlying_price</th>
      <th>underlying_index</th>
      <th>timestamp</th>
      <th>stats</th>
      <th>state</th>
      <th>settlement_price</th>
      <th>open_interest</th>
      <th>min_price</th>
      <th>max_price</th>
      <th>mark_price</th>
      <th>...</th>
      <th>change_id</th>
      <th>bids</th>
      <th>bid_iv</th>
      <th>best_bid_price</th>
      <th>best_bid_amount</th>
      <th>best_ask_price</th>
      <th>best_ask_amount</th>
      <th>asks</th>
      <th>ask_iv</th>
      <th>option_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34507.37</td>
      <td>BTC-24SEP21</td>
      <td>1625478574014</td>
      <td>{'volume': 0.1, 'price_change': 0.0, 'low': 0....</td>
      <td>open</td>
      <td>0.27</td>
      <td>539.1</td>
      <td>0.2040</td>
      <td>0.3635</td>
      <td>0.271508</td>
      <td>...</td>
      <td>32733059720</td>
      <td>[[0.267, 0.6], [0.2665, 0.6], [0.2315, 1.0], [...</td>
      <td>87.60</td>
      <td>0.2670</td>
      <td>0.6</td>
      <td>0.2890</td>
      <td>0.3</td>
      <td>[[0.289, 0.3], [0.2895, 20.2], [0.3115, 1.0], ...</td>
      <td>99.40</td>
      <td>P</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36300.52</td>
      <td>BTC-24JUN22</td>
      <td>1625478574014</td>
      <td>{'volume': None, 'price_change': None, 'low': ...</td>
      <td>open</td>
      <td>0.41</td>
      <td>0.0</td>
      <td>0.3385</td>
      <td>0.5140</td>
      <td>0.414394</td>
      <td>...</td>
      <td>32733056884</td>
      <td>[[0.374, 12.0], [0.3405, 1.0], [0.202, 0.4], [...</td>
      <td>79.92</td>
      <td>0.3740</td>
      <td>12.0</td>
      <td>0.4570</td>
      <td>12.0</td>
      <td>[[0.457, 12.0], [0.4905, 1.0]]</td>
      <td>106.33</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34222.56</td>
      <td>BTC-9JUL21</td>
      <td>1625478574014</td>
      <td>{'volume': 1.1, 'price_change': -21.978, 'low'...</td>
      <td>open</td>
      <td>0.10</td>
      <td>19.7</td>
      <td>0.0680</td>
      <td>0.1480</td>
      <td>0.103744</td>
      <td>...</td>
      <td>32733058155</td>
      <td>[[0.1025, 3.0], [0.1015, 1.7], [0.0975, 0.4], ...</td>
      <td>97.46</td>
      <td>0.1025</td>
      <td>3.0</td>
      <td>0.1055</td>
      <td>3.0</td>
      <td>[[0.1055, 3.0], [0.106, 0.7], [0.1065, 1.0], [...</td>
      <td>109.16</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 27 columns</p>
</div>



La donnée brute sur les options contient beaucoup d'information, elles seront traitées en amont dans les modèles.

### Initialisation des modèles

Les modèles s'initialisent de la façon suivante, avec deux paramètres, le dataframe brut scrapé, ainsi que le type d'option à garder : $$\tt{ \{C : Call, P : Put, B : Both\} }$$


```python
BS = BlackScholes(raw_df.copy(), 'B')
M = Merton(raw_df.copy(), 'B')
H = Heston(raw_df.copy(), 'B')
```

Chaque modèle possède désormais un DataFrame $\tt{df}$ trié lors de l'initialisation des modèles par la pipeline de tri du module $\tt{utils}$, il ne contient que les informations utiles pour la suite.


```python
BS.df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S</th>
      <th>K</th>
      <th>V</th>
      <th>_T</th>
      <th>bids</th>
      <th>asks</th>
      <th>last_price</th>
      <th>mark_price</th>
      <th>option_type</th>
      <th>mid_iv</th>
      <th>mark_iv</th>
      <th>mid</th>
      <th>moneyness</th>
      <th>I_VOL</th>
      <th>IV_moneyness</th>
      <th>BS_PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>213</th>
      <td>34274.65</td>
      <td>30000.0</td>
      <td>13.4</td>
      <td>0.001389</td>
      <td>0.000000</td>
      <td>0.002133</td>
      <td>0.0005</td>
      <td>0.000202</td>
      <td>P</td>
      <td>65.625</td>
      <td>115.39</td>
      <td>0.001066</td>
      <td>1.142488</td>
      <td>0.811455</td>
      <td>42238.520998</td>
      <td>0.001064</td>
    </tr>
    <tr>
      <th>27</th>
      <td>34275.83</td>
      <td>31000.0</td>
      <td>20.8</td>
      <td>0.001389</td>
      <td>0.000500</td>
      <td>0.002695</td>
      <td>0.0010</td>
      <td>0.000930</td>
      <td>P</td>
      <td>115.080</td>
      <td>115.25</td>
      <td>0.001598</td>
      <td>1.105672</td>
      <td>0.631219</td>
      <td>54301.027215</td>
      <td>0.001593</td>
    </tr>
    <tr>
      <th>24</th>
      <td>34275.83</td>
      <td>32000.0</td>
      <td>15.2</td>
      <td>0.001389</td>
      <td>0.001279</td>
      <td>0.003725</td>
      <td>0.0020</td>
      <td>0.002008</td>
      <td>P</td>
      <td>100.650</td>
      <td>101.15</td>
      <td>0.002502</td>
      <td>1.071120</td>
      <td>0.449373</td>
      <td>76274.719335</td>
      <td>0.002492</td>
    </tr>
    <tr>
      <th>388</th>
      <td>34273.54</td>
      <td>33000.0</td>
      <td>28.5</td>
      <td>0.001389</td>
      <td>0.003552</td>
      <td>0.007324</td>
      <td>0.0040</td>
      <td>0.004897</td>
      <td>P</td>
      <td>89.910</td>
      <td>89.21</td>
      <td>0.005438</td>
      <td>1.038592</td>
      <td>0.266652</td>
      <td>128532.664844</td>
      <td>0.005406</td>
    </tr>
    <tr>
      <th>227</th>
      <td>34274.33</td>
      <td>34000.0</td>
      <td>53.6</td>
      <td>0.001389</td>
      <td>0.010437</td>
      <td>0.015253</td>
      <td>0.0115</td>
      <td>0.012091</td>
      <td>P</td>
      <td>78.445</td>
      <td>78.93</td>
      <td>0.012845</td>
      <td>1.008069</td>
      <td>0.066302</td>
      <td>516945.232329</td>
      <td>0.012577</td>
    </tr>
  </tbody>
</table>
</div>



La méthode $\tt{initialize}$ permet d'initialiser les paramètres propres à chaque modèle, depuis un vecteur $\theta$, le booléen $\tt{reset}$ permet de choisir de réinitialiser aux derniers paramètres calibrés enregistrés.


```python
BS.initialize(reset = True)
M.initialize(reset = True)
H.initialize(reset = True)
```

### Ajouts des volatiliés implicites

La méthode append_imp_vol_to_df ajoute une colonne au DataFrame du modèle avec les volatilités implicites, calculées  dans le module $\tt{common\_all}$


```python
BS.append_imp_vol_to_df()
M.append_imp_vol_to_df()
H.append_imp_vol_to_df()
```

### Ajouts des Prix respectifs de chaque modèles aux df 

Chaque modèle possède une méthode $\tt{Price}$ qui calcule le prix d'une option sous ce modèle.


```python
S = 34205.37
K = 35000.0
T = 0.067142
v = 0.028870
Flag = 'C'
print("Black-Scholes Price : ${}".format(S*BS.Price(S=S, K=K, T=T, sigma=v, CallPutFlag=Flag)))
print("Merton Price        : ${}".format(S*M.Price(S=S, K=K, T=T, sigma=v, CallPutFlag=Flag)))
print("Heston Price        : ${}".format(S*H.Price(S=S, K=K, T=T, sigma=v, CallPutFlag=Flag)))
print("Market Price        : ${}".format(S*0.077089))
```

    Black-Scholes Price : $3626.812075362676
    Merton Price        : $3629.7543747209543
    Heston Price        : $2996.9214575768083
    Market Price        : $2636.8577679300006


La vectorisation de cette fonction et son application aux lignes du DataFrame permet de calculer une liste de prix qui pourront être comparés avec ceux du marché. Cela se fait via la méthode $\tt{append\_price}$.


```python
BS.append_price()
M.append_price()
H.append_price()

BS.df[['BS_PRICE','mid']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BS_PRICE</th>
      <th>mid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>213</th>
      <td>0.001064</td>
      <td>0.001066</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.001593</td>
      <td>0.001598</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.002492</td>
      <td>0.002502</td>
    </tr>
    <tr>
      <th>388</th>
      <td>0.005406</td>
      <td>0.005438</td>
    </tr>
    <tr>
      <th>227</th>
      <td>0.012577</td>
      <td>0.012845</td>
    </tr>
  </tbody>
</table>
</div>




```python
bad_prices_indexes = H.df[H.df.mid >= 1].index


BS.df = BS.df.drop(bad_prices_indexes)
H.df = H.df.drop(bad_prices_indexes)
M.df = M.df.drop(bad_prices_indexes)

bad_prices_indexes =  BS.df[BS.df.BS_PRICE >= 1].index

BS.df = BS.df.drop(bad_prices_indexes)
H.df = H.df.drop(bad_prices_indexes)
M.df = M.df.drop(bad_prices_indexes)



```


```python
#H.df[H.df.mid >= 0].index
```

### Optimisation des paramètres (pour Merton et Heston)

L'optimisation des paramètres (pour les Modèles de Merton et de Heston) se fait part la méthode $\verb+optimize+$ des classes. On peut choisir de partir d'un vecteur de paramètre initial $x_0$, sinon, un vecteur aléatoire dans les bornes est choisi.


```python
H.optimize(x0 = None, tol = 1e-4, max_iter =300, update_when_done=True )
```




    array([0.90260584, 0.42127592, 0.13607334, 0.37337941, 0.20804514])




```python
M.optimize(tol = 1e-7, max_iter=500, update_when_done=True)
```




    array([9.17579001e-01, 3.50404358e-02, 2.89575783e-04])




```python
from sklearn.metrics import mean_absolute_error

x = BS.df['K']
y = BS.df['_T']
z = BS.df['mid']

x1 = BS.df['K']
y1 = BS.df['_T']
z1 = BS.df['BS_PRICE']

x2 = M.df['K']
y2 = M.df['_T']
z2 = M.df['MERTON_PRICE']

x3 = H.df['K']
y3 = H.df['_T']
z3 = H.df['HESTON_PRICE']




fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z , mode='markers', name='Real Market Price', marker=dict(opacity=0.8)),
                      go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', name='BlackScholes Model Price', marker=dict(opacity=0.8)),
                      go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', name='Merton Model Price', marker=dict(opacity=0.8),),
                      go.Scatter3d(x=x3, y=y3, z=z3, mode='markers', name='Heston Model Price', marker=dict(opacity=0.8))])


fig.update_scenes(xaxis_title_text='Strike', yaxis_title_text='Exp', zaxis_title_text='Price') 
fig.show()

#rms_merton = mean_squared_error(z2, z3, squared=False)


```


<div>                            <div id="3d0a502d-aa73-4d85-bbbe-9289d56b41b4" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("3d0a502d-aa73-4d85-bbbe-9289d56b41b4")) {                    Plotly.newPlot(                        "3d0a502d-aa73-4d85-bbbe-9289d56b41b4",                        [{"marker": {"opacity": 0.8}, "mode": "markers", "name": "Real Market Price", "type": "scatter3d", "x": [30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 34000.0, 36000.0, 26000.0, 28000.0, 29000.0, 30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 40000.0, 42000.0, 24000.0, 26000.0, 28000.0, 30000.0, 31000.0, 32000.0, 34000.0, 35000.0, 36000.0, 38000.0, 40000.0, 42000.0, 45000.0, 28000.0, 30000.0, 32000.0, 34000.0, 38000.0, 40000.0, 42000.0, 20000.0, 25000.0, 26000.0, 28000.0, 30000.0, 32000.0, 34000.0, 35000.0, 36000.0, 40000.0, 45000.0, 50000.0, 60000.0, 25000.0, 30000.0, 35000.0, 45000.0, 50000.0, 55000.0, 60000.0, 70000.0, 24000.0, 30000.0, 32000.0, 36000.0, 44000.0, 48000.0, 56000.0, 60000.0, 72000.0], "y": [0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.004128463878804267, 0.004128463878804267, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784], "z": [0.00106640625, 0.0015975029726516053, 0.00250160630036899, 0.005438276429665824, 0.012844996780068625, 0.007454539667527555, 0.002113006801594046, 0.0011703747072599531, 0.000974406991260924, 0.02068742332643704, 0.00512750367341005, 0.001881867290948667, 0.006806065328422077, 0.004384890656063618, 0.023061798970185978, 0.009409976617905064, 0.014131033155892515, 0.02045675055200624, 0.03599373442691029, 0.023473380521087783, 0.021528198676730965, 0.0071913557058185165, 0.005450479457550956, 0.00203290456351785, 0.0012492187500000001, 0.003623168963745109, 0.006543402979383353, 0.010986273946360155, 0.01918148303649224, 0.02511588853999948, 0.0329126755664284, 0.05498632320274134, 0.04708342125872908, 0.035405771146842575, 0.01840713757931892, 0.009714135349654082, 0.005161639821594033, 0.0025746314886314885, 0.020349024607930637, 0.030731136166522118, 0.04610048864403918, 0.07122154268145647, 0.03127491267046315, 0.01901332808687696, 0.01100754578754579, 0.006258257353797559, 0.03305626282530216, 0.02991785526647038, 0.040852900130171614, 0.042799865709842705, 0.09316088639095375, 0.08768332322630651, 0.07708873307543523, 0.06347444832062638, 0.03250354128432234, 0.011230642974255071, 0.009081392845841887, 0.003139099144434066, 0.03722468166117389, 0.09716253352795139, 0.125332403662385, 0.04175280414150129, 0.030905656665678224, 0.019573827274780638, 0.011731439053528286, 0.006615955578473444, 0.061015719974312224, 0.10321091322582551, 0.1419917839645385, 0.15075195866482186, 0.08230751071878334, 0.06781079446618832, 0.049930760521215145, 0.03761489852604219, 0.026930253906250003]}, {"marker": {"opacity": 0.8}, "mode": "markers", "name": "BlackScholes Model Price", "type": "scatter3d", "x": [30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 34000.0, 36000.0, 26000.0, 28000.0, 29000.0, 30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 40000.0, 42000.0, 24000.0, 26000.0, 28000.0, 30000.0, 31000.0, 32000.0, 34000.0, 35000.0, 36000.0, 38000.0, 40000.0, 42000.0, 45000.0, 28000.0, 30000.0, 32000.0, 34000.0, 38000.0, 40000.0, 42000.0, 20000.0, 25000.0, 26000.0, 28000.0, 30000.0, 32000.0, 34000.0, 35000.0, 36000.0, 40000.0, 45000.0, 50000.0, 60000.0, 25000.0, 30000.0, 35000.0, 45000.0, 50000.0, 55000.0, 60000.0, 70000.0, 24000.0, 30000.0, 32000.0, 36000.0, 44000.0, 48000.0, 56000.0, 60000.0, 72000.0], "y": [0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.0013887378514070067, 0.004128463878804267, 0.004128463878804267, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.009607915933598787, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.02878599812537961, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.04796408031716043, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.06714216250894126, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.14385449127606453, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784, 0.22056682004318784], "z": [0.0010640453052885301, 0.0015930390288502738, 0.002492245849659236, 0.005405844086069145, 0.012576789440423397, 0.007528169155310849, 0.0021240223386335133, 0.0011746295983148802, 0.000977126739539036, 0.019439280025892458, 0.005199709614374104, 0.0018677907116020354, 0.0067454995604084855, 0.004336464981821719, 0.02279529092540189, 0.009256341628949283, 0.013819013451717765, 0.01971423238308212, 0.03079795661522411, 0.024795068893549654, 0.02212324373051011, 0.007343620382939786, 0.005541130592706001, 0.0020586967663431266, 0.001262121110503639, 0.0035627088505742477, 0.0064122422167733895, 0.01070812581899716, 0.018510542482675252, 0.024019053586022743, 0.030957617714060248, 0.034662043865139935, 0.05454504534535687, 0.03821712243299835, 0.019227100673718578, 0.010036478145613703, 0.005303361381648941, 0.002632412232394049, 0.019546961113336536, 0.029048035044113796, 0.04179722920182094, 0.03381713628603933, 0.03349148654109779, 0.02000400986336981, 0.01148133808979801, 0.006100208921709538, 0.031895613538207135, 0.028723154833441455, 0.038768679753529156, 0.039666338892645925, 0.0821639672654868, 0.03173945975468939, 0.10601295651874665, 0.07487704410469576, 0.03476289394316945, 0.011750718808115068, 0.009400260264498472, 0.0032230737507576435, 0.03454251722119661, 0.08412507249784085, 0.2643973932196957, 0.04551728519025833, 0.03302888659917547, 0.020714885872078148, 0.012345017767610267, 0.006906721704453889, 0.055395020736511036, 0.08336739614449762, 0.09974880936331942, 0.2634034277075159, 0.09418050987813853, 0.07532957512907457, 0.05394793369185402, 0.040368233997740255, 0.028499860144139855]}, {"marker": {"opacity": 0.8}, "mode": "markers", "name": "Merton Model Price", "type": "scatter3d", "x": [30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 34000.0, 36000.0, 26000.0, 28000.0, 29000.0, 30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 40000.0, 42000.0, 24000.0, 26000.0, 28000.0, 30000.0, 31000.0, 32000.0, 34000.0, 35000.0, 36000.0, 38000.0, 40000.0, 42000.0, 45000.0, 28000.0, 30000.0, 32000.0, 34000.0, 38000.0, 40000.0, 42000.0, 20000.0, 25000.0, 26000.0, 28000.0, 30000.0, 32000.0, 34000.0, 35000.0, 36000.0, 40000.0, 45000.0, 50000.0, 60000.0, 25000.0, 30000.0, 35000.0, 45000.0, 50000.0, 55000.0, 60000.0, 70000.0, 24000.0, 30000.0, 32000.0, 36000.0, 44000.0, 48000.0, 56000.0, 60000.0, 72000.0], "y": [0.0013887372493350766, 0.0013887372493350766, 0.0013887372493350766, 0.0013887372493350766, 0.0013887372493350766, 0.0013887372493350766, 0.0013887372493350766, 0.0013887372493350766, 0.0013887372493350766, 0.0041284632767323364, 0.0041284632767323364, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.009607915331526858, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.02878599752330768, 0.0479640797150885, 0.0479640797150885, 0.0479640797150885, 0.0479640797150885, 0.0479640797150885, 0.0479640797150885, 0.0479640797150885, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.06714216190686932, 0.1438544906739926, 0.1438544906739926, 0.1438544906739926, 0.1438544906739926, 0.1438544906739926, 0.1438544906739926, 0.1438544906739926, 0.1438544906739926, 0.2205668194411159, 0.2205668194411159, 0.2205668194411159, 0.2205668194411159, 0.2205668194411159, 0.2205668194411159, 0.2205668194411159, 0.2205668194411159, 0.2205668194411159], "z": [0.0011098811055146895, 0.0017259702557893496, 0.002819495975704286, 0.0060499260224047245, 0.013603766549196752, 0.007528490756938666, 0.0021240551771268654, 0.0011746361330985918, 0.000977215528658218, 0.022508081653469993, 0.005199920827263979, 0.0018749010008699857, 0.0067993398737456155, 0.004466679659023848, 0.02320437560823261, 0.010263189614474465, 0.016192267221173135, 0.024290579619319314, 0.03803881669376762, 0.024799311988493805, 0.022124748264955484, 0.0073439842989541666, 0.005541336965905451, 0.002058752707171805, 0.0012621480733547683, 0.003569791595270244, 0.006444326146421165, 0.010886091435852825, 0.019720997660683685, 0.02713169185900596, 0.038150035806133034, 0.056379651890319946, 0.05456695882540979, 0.03822419437531705, 0.019228975598182697, 0.01003718654930247, 0.005303664902667771, 0.0026325314731148754, 0.019881636114179663, 0.031125795835020807, 0.053746464978069765, 0.06991743924981675, 0.03349658342348474, 0.020006191884540118, 0.011482349800237294, 0.0061038430299968065, 0.03196936083289003, 0.02884837055580108, 0.03931064283111524, 0.042631851617813035, 0.09890780400819284, 0.0821549488379146, 0.10609895868468718, 0.074906188482599, 0.03476786900066562, 0.011751786983358594, 0.009400873975851824, 0.003223223976793189, 0.034695759631939105, 0.09011637267709922, 0.26487212669462273, 0.04552498843459966, 0.03303293376383646, 0.020716958322521527, 0.012346093037854616, 0.006907199175577824, 0.05556733682804994, 0.09143546152797219, 0.14305963279733658, 0.2637404264746402, 0.0942054274673325, 0.07534428836354885, 0.053955022383332316, 0.04037293157754549, 0.028502298687971033]}, {"marker": {"opacity": 0.8}, "mode": "markers", "name": "Heston Model Price", "type": "scatter3d", "x": [30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 34000.0, 36000.0, 26000.0, 28000.0, 29000.0, 30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 40000.0, 42000.0, 24000.0, 26000.0, 28000.0, 30000.0, 31000.0, 32000.0, 34000.0, 35000.0, 36000.0, 38000.0, 40000.0, 42000.0, 45000.0, 28000.0, 30000.0, 32000.0, 34000.0, 38000.0, 40000.0, 42000.0, 20000.0, 25000.0, 26000.0, 28000.0, 30000.0, 32000.0, 34000.0, 35000.0, 36000.0, 40000.0, 45000.0, 50000.0, 60000.0, 25000.0, 30000.0, 35000.0, 45000.0, 50000.0, 55000.0, 60000.0, 70000.0, 24000.0, 30000.0, 32000.0, 36000.0, 44000.0, 48000.0, 56000.0, 60000.0, 72000.0], "y": [0.0013887367104286473, 0.0013887367104286473, 0.0013887367104286473, 0.0013887367104286473, 0.0013887367104286473, 0.0013887367104286473, 0.0013887367104286473, 0.0013887367104286473, 0.0013887367104286473, 0.004128462737825907, 0.004128462737825907, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.009607914792620428, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.02878599698440125, 0.04796407917618207, 0.04796407917618207, 0.04796407917618207, 0.04796407917618207, 0.04796407917618207, 0.04796407917618207, 0.04796407917618207, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.06714216136796289, 0.1438544901350862, 0.1438544901350862, 0.1438544901350862, 0.1438544901350862, 0.1438544901350862, 0.1438544901350862, 0.1438544901350862, 0.1438544901350862, 0.22056681890220947, 0.22056681890220947, 0.22056681890220947, 0.22056681890220947, 0.22056681890220947, 0.22056681890220947, 0.22056681890220947, 0.22056681890220947, 0.22056681890220947], "z": [-3.482175873446541e-06, 2.698614257492667e-05, 0.0003341361877754202, 0.002532516323624171, 0.010420173040324924, 0.00612911358258647, 0.0013711970182635853, 0.00020019790935834055, 2.3214092104927084e-05, 0.020673275921863618, 0.007335637812057553, 3.600907404528984e-05, 0.000463040732215634, 0.0012842382334868115, 0.0030955351018238137, 0.006549264798910891, 0.012401094110079308, 0.0213711532514893, 0.033950657626309215, 0.02720110567957866, 0.017592089388660064, 0.010845429904068516, 0.006399697422889363, 0.0019526411379967158, 0.0005094696193635778, 0.000643168223111318, 0.002510248642224519, 0.007416670825170271, 0.017583300825195772, 0.025297462353479516, 0.03506508593923729, 0.06101797575150664, 0.05392075645312351, 0.04288232482119912, 0.026185567779408597, 0.015306192599706196, 0.008587555232105089, 0.0033744486821265522, 0.016620676681673274, 0.030979305263235613, 0.05181110961599917, 0.0794729953020881, 0.04276358933616377, 0.029234977664505755, 0.019601723365336346, 0.0009220580911716292, 0.009876384527893172, 0.01401096360808975, 0.025745515181599475, 0.04280283578649702, 0.06567887061860425, 0.09440333827405176, 0.08761564206563331, 0.07615619053782371, 0.041997273324643175, 0.018690386906657055, 0.00787518649347629, 0.0012657317976001645, 0.03083293330526389, 0.07884171979634885, 0.13418598150394842, 0.05312166119670807, 0.03274585841439036, 0.020051592797766655, 0.01226357583985164, 0.004594755923169263, 0.0416816942549544, 0.1058399231802122, 0.13438047610133286, 0.15770290528875372, 0.09027002057298221, 0.06807850637766877, 0.03876141801112666, 0.029291611013202174, 0.0128910398553463]}],                        {"scene": {"xaxis": {"title": {"text": "Strike"}}, "yaxis": {"title": {"text": "Exp"}}, "zaxis": {"title": {"text": "Price"}}}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('3d0a502d-aa73-4d85-bbbe-9289d56b41b4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


La fonction $\tt{Eval\_Metrics}$ du module $\tt{utils}$ permet de calculer quelques métriques d'évaluations pour nos 3 modèles.
- RMSE, sensible aux gros écarts
- MAE, sensible aux plus petits écarts
- $R^2Score$ est le coefficient de détermination


```python
Eval_Metrics(BS=BS,M=M,H=H)
```

    =============== Root Mean Squared Error ================
    
    B&S    = 2.334 %
    MERTON    = 2.128 %
    HESTON    = 0.821 % 
    
    =============== Mean Absolute Error ================
    
    B&S    = 0.778 %
    MERTON    = 0.566 %
    HESTON    = 0.579 % 
    
    =============== R2 Score ================
    
    B&S    = 74.244 %
    MERTON    = 80.441 %
    HESTON    = 94.433 % 
    



```python

```


```python

```
