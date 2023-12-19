import numpy as np
import pandas as pd

base_path = ".cache/data/"


def build_reg_data(data_name="community"):
    if data_name == "community":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        attrib = pd.read_csv(base_path + 'communities_attributes.csv', delim_whitespace=True)
        data = pd.read_csv(base_path + 'communities.data', names=attrib['attributes'])
        data = data.drop(columns=['state', 'county',
                                  'community', 'communityname',
                                  'fold'], axis=1)

        data = data.replace('?', np.nan)

        # Impute mean values for samples with missing values

        # imputer = SimpleImputer(missing_values = 'NaN', strategy = 'mean')

        # imputer = imputer.fit(data[['OtherPerCap']])
        # data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
        data['OtherPerCap'] = data['OtherPerCap'].astype("float")
        mean_value = data['OtherPerCap'].mean()
        data['OtherPerCap'].fillna(value=mean_value, inplace=True)
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values

            
        # imputer = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
        
        # imputer = imputer.fit(data[['OtherPerCap']])
        # data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
        data['OtherPerCap'] = data['OtherPerCap'].astype("float")
        mean_value = data['OtherPerCap'].mean()
        data['OtherPerCap'].fillna(value=mean_value, inplace=True)
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values
    elif data_name == "S&P500":
        data = pd.read_csv(base_path + "S&P500.csv")
        # the price of stock index
        prices = np.array(data["Close/Last"])
        days = prices.shape[0]
        # the rate of return from 2-ed day
        rate_return = (prices[1:days] - prices[:days-1])/prices[:days-1]  
        volatility = rate_return**2
        
        sigma2 =  rate_return/ np.random.randn(rate_return.shape[0])
        X = np.concatenate((volatility[:-1], sigma2[:-1]))
        y = sigma2[1:]
    elif data_name == "synthetic":
        X = np.random.rand(500,5)
        y_wo_noise = 10*np.sin(X[:,0]*X[:,1]*np.pi) +20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4]  
        eplison = np.zeros(500)
        phi = theta = 0.8
        delta_t_1 = np.random.randn() 
        for i in range(1,500):
            delta_t = np.random.randn() 
            eplison[i] = phi*eplison[i-1] + delta_t_1 + theta * delta_t
            delta_t_1 =  delta_t
            
        y = y_wo_noise + eplison


    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y
