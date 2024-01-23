spy_neu = spy_neu_temp.copy(deep=True)
spy_neu = spy_neu[['date', 'spy_neu_price',]]

# Set the 'date_time' column as the index
spy_neu.set_index('date', inplace=True)

# Shift the 'price' column by 5 minutes
spy_neu['price_5_min_ago'] = spy_neu['spy_neu_price'].shift(freq='5T')

# Calculate the difference
#spy_neu['diff'] = spy_neu['spy_neu_price'] - spy_neu['price_5_min_ago']
spy_neu['diff'] = 100 * np.log10(spy_neu['spy_neu_price'] / spy_neu['price_5_min_ago'])

#For diff remove observations that are below the 5th percentile or above the 95th percentile
spy_neu = spy_neu[spy_neu['diff'] > spy_neu['diff'].quantile(.05)]
spy_neu = spy_neu[spy_neu['diff'] < spy_neu['diff'].quantile(.95)]

# Reset the index if you want 'date_time' back as a column
spy_neu.reset_index(inplace=True)

spy_neu = spy_neu[['date', 'spy_neu_price', 'diff']]
spy_neu['time'] = spy_neu['date'].dt.time
spy_neu_2 = spy_neu.groupby(['time']).agg(["mean", "var"]).reset_index()
spy_neu_2.columns = ['_'.join(col).strip() for col in spy_neu_2.columns.values]
spy_neu_2['spy_neu_price_sd'] = spy_neu_2['spy_neu_price_var'] ** 0.5
spy_neu_2['diff_sd'] = spy_neu_2['diff_var'] ** 0.5
spy_neu_2 = spy_neu_2[['time_', 'spy_neu_price_mean', 'spy_neu_price_sd', 'diff_mean', 'diff_sd']]
spy_neu_2 = spy_neu_2.rename(columns={'time_': 'time'})
spy_neu_2 = spy_neu_2.dropna()
print(spy_neu_2.head())

spy_neu_2['time'] = pd.to_datetime(spy_neu_2['time'], format='%H:%M:%S')
print(spy_neu_2.dtypes) 
spy_neu_2 = spy_neu_2.sort_values(by=['time'])
spy_neu_2 = spy_neu_2.set_index('time')
spy_neu_2 = spy_neu_2.between_time('12:00:00', '16:00:00')
spy_neu_2 = spy_neu_2.reset_index()
print(spy_neu_2.head())