import deribit_data as dm
btc_data = dm.Options("BTC")


df = btc_data.collect_data(save_csv=False)
#df[['instrument_name', 'last_price', 'mark_iv', 'open_interest']].head()

print(df.head())