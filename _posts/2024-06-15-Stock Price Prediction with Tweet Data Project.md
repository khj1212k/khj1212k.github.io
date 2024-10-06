---
layout: single
title: "Stock Price Prediction with Tweet Data Project"
categories: Projects
---
# 0. import

```python
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM,Conv1D,Conv2D,MaxPooling2D,MaxPooling1D,Flatten,Bidirectional
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM,Bidirectional

matplotlib.rcParams['font.family'] = ['Heiti TC']
pd.set_option('display.float_format', '{:.3f}'.format)

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
  
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
  
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)
```

# 1. Read CSVs

```python
Stock_price_li = os.listdir('data/Stock_price')
Tweet_li = os.listdir('data/Tweet')
print(Stock_price_li)
print(Tweet_li)
```

['AMZN.csv', 'MSFT.csv', 'TSLA.csv', 'GOOGL.csv', 'GOOG.csv', 'AAPL.csv']
['Company.csv', 'Company_Tweet.csv', 'Tweet.csv']

```python
AAPL = pd.read_csv('data/Stock_price/AAPL.csv')
AMZN = pd.read_csv('data/Stock_price/AMZN.csv')
GOOG = pd.read_csv('data/Stock_price/GOOG.csv')
GOOGL = pd.read_csv('data/Stock_price/GOOGL.csv')
MSFT = pd.read_csv('data/Stock_price/MSFT.csv')
TSLA = pd.read_csv('data/Stock_price/TSLA.csv')
AAPL['ticker_symbol'] = 'AAPL'
AMZN['ticker_symbol'] = 'AMZN'
GOOG['ticker_symbol'] = 'GOOG'
GOOGL['ticker_symbol'] = 'GOOGL'
MSFT['ticker_symbol'] = 'MSFT'
TSLA['ticker_symbol'] = 'TSLA'
display('AAPL','AMZN','GOOG','GOOGL','MSFT','TSLA')
```

<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>AAPL</p><div>
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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
      <th>ticker_symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-12-28</td>
      <td>194.140</td>
      <td>194.660</td>
      <td>193.170</td>
      <td>193.580</td>
      <td>193.580</td>
      <td>34014500</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-12-27</td>
      <td>192.490</td>
      <td>193.500</td>
      <td>191.090</td>
      <td>193.150</td>
      <td>193.150</td>
      <td>48087700</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-12-26</td>
      <td>193.610</td>
      <td>193.890</td>
      <td>192.830</td>
      <td>193.050</td>
      <td>193.050</td>
      <td>28919300</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-12-22</td>
      <td>195.180</td>
      <td>195.410</td>
      <td>192.970</td>
      <td>193.600</td>
      <td>193.600</td>
      <td>37122800</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-12-21</td>
      <td>196.100</td>
      <td>197.080</td>
      <td>193.500</td>
      <td>194.680</td>
      <td>194.680</td>
      <td>46482500</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10847</th>
      <td>1980-12-18</td>
      <td>0.475</td>
      <td>0.478</td>
      <td>0.475</td>
      <td>0.475</td>
      <td>0.377</td>
      <td>18362400</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>10848</th>
      <td>1980-12-17</td>
      <td>0.462</td>
      <td>0.464</td>
      <td>0.462</td>
      <td>0.462</td>
      <td>0.366</td>
      <td>21610400</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>10849</th>
      <td>1980-12-16</td>
      <td>0.453</td>
      <td>0.453</td>
      <td>0.451</td>
      <td>0.451</td>
      <td>0.357</td>
      <td>26432000</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>10850</th>
      <td>1980-12-15</td>
      <td>0.489</td>
      <td>0.489</td>
      <td>0.487</td>
      <td>0.487</td>
      <td>0.386</td>
      <td>43971200</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>10851</th>
      <td>1980-12-12</td>
      <td>0.513</td>
      <td>0.516</td>
      <td>0.513</td>
      <td>0.513</td>
      <td>0.407</td>
      <td>117258400</td>
      <td>AAPL</td>
    </tr>
  </tbody>
</table>
<p>10852 rows × 8 columns</p>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>AMZN</p><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


  
```python
plt.figure(figsize=(14, 7))
sns.lineplot(data=stock_data, x='date', y='close', hue='ticker_symbol')
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(title='Ticker Symbol')
plt.show()
```

![png](tweet_stock_files/tweet_stock_13_0.png)

```python
# Convert post_date in Tweet to match the date format in stock_data
Tweet['date'] = Tweet['post_date'].dt.date
Tweet['date'] = pd.to_datetime(Tweet['date'])

# Merge Tweet with Company to get ticker symbols
Tweet = Tweet.merge(Company_Tweet, how='left', on = 'tweet_id')
Tweet.head()
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
      <th>tweet_id</th>
      <th>writer</th>
      <th>post_date</th>
      <th>body</th>
      <th>comment_num</th>
      <th>retweet_num</th>
      <th>like_num</th>
      <th>date</th>
      <th>ticker_symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550441509175443456</td>
      <td>VisualStockRSRC</td>
      <td>2015-01-01 00:00:57</td>
      <td>lx21 made $10,008  on $AAPL -Check it out! htt...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>550441672312512512</td>
      <td>KeralaGuy77</td>
      <td>2015-01-01 00:01:36</td>
      <td>Insanity of today weirdo massive selling. $aap...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>550441732014223360</td>
      <td>DozenStocks</td>
      <td>2015-01-01 00:01:50</td>
      <td>S&P100 #Stocks Performance $HD $LOW $SBUX $TGT...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-01-01</td>
      <td>AMZN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>550442977802207232</td>
      <td>ShowDreamCar</td>
      <td>2015-01-01 00:06:47</td>
      <td>$GM $TSLA: Volkswagen Pushes 2014 Record Recal...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>TSLA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>550443807834402816</td>
      <td>i_Know_First</td>
      <td>2015-01-01 00:10:05</td>
      <td>Swing Trading: Up To 8.91% Return In 14 Days h...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
    </tr>
  </tbody>
</table>
</div>

```python
stock_data.describe()
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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>39283</td>
      <td>39283.000</td>
      <td>39283.000</td>
      <td>39283.000</td>
      <td>39283.000</td>
      <td>39283.000</td>
      <td>39283.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2008-03-28 20:02:29.927449856</td>
      <td>206.741</td>
      <td>209.017</td>
      <td>204.351</td>
      <td>206.768</td>
      <td>205.060</td>
      <td>45274189.005</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1980-12-12 00:00:00</td>
      <td>0.089</td>
      <td>0.092</td>
      <td>0.089</td>
      <td>0.090</td>
      <td>0.058</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2001-02-15 00:00:00</td>
      <td>12.376</td>
      <td>12.613</td>
      <td>12.151</td>
      <td>12.340</td>
      <td>10.520</td>
      <td>5998450.000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2010-05-05 00:00:00</td>
      <td>75.563</td>
      <td>76.630</td>
      <td>74.574</td>
      <td>75.710</td>
      <td>71.963</td>
      <td>26844000.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2016-11-10 00:00:00</td>
      <td>244.339</td>
      <td>247.500</td>
      <td>241.180</td>
      <td>244.347</td>
      <td>243.779</td>
      <td>62173100.000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2023-12-28 00:00:00</td>
      <td>2912.010</td>
      <td>2955.560</td>
      <td>2871.100</td>
      <td>2890.300</td>
      <td>2890.300</td>
      <td>1855410200.000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>338.138</td>
      <td>341.476</td>
      <td>334.644</td>
      <td>338.247</td>
      <td>338.988</td>
      <td>60113427.380</td>
    </tr>
  </tbody>
</table>
</div>

```python
Tweet.describe()
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
      <th>tweet_id</th>
      <th>post_date</th>
      <th>comment_num</th>
      <th>retweet_num</th>
      <th>like_num</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4336445.000</td>
      <td>4336445</td>
      <td>4336445.000</td>
      <td>4336445.000</td>
      <td>4336445.000</td>
      <td>4336445</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>883428183858007296.000</td>
      <td>2017-07-07 20:51:06.490715136</td>
      <td>0.292</td>
      <td>0.635</td>
      <td>2.104</td>
      <td>2017-07-07 06:25:10.584635904</td>
    </tr>
    <tr>
      <th>min</th>
      <td>550441509175443456.000</td>
      <td>2015-01-01 00:00:57</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2015-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>718545015031488512.000</td>
      <td>2016-04-08 21:04:16</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2016-04-08 00:00:00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>882593382354309120.000</td>
      <td>2017-07-05 13:33:54</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2017-07-05 00:00:00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1054775619219673088.000</td>
      <td>2018-10-23 16:44:39</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>2018-10-23 00:00:00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1212160477159206912.000</td>
      <td>2019-12-31 23:55:53</td>
      <td>631.000</td>
      <td>999.000</td>
      <td>999.000</td>
      <td>2019-12-31 00:00:00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>192773549694689952.000</td>
      <td>NaN</td>
      <td>1.886</td>
      <td>6.986</td>
      <td>13.717</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

```python
stock_data.date.min(),stock_data.date.max()
```

(Timestamp('1980-12-12 00:00:00'), Timestamp('2023-12-28 00:00:00'))

```python
Tweet.post_date.min(),Tweet.post_date.max()
```

(Timestamp('2015-01-01 00:00:57'), Timestamp('2019-12-31 23:55:53'))
`주식 데이터는 시간 범위가 1980부터 2023까지 넓은데 트위터 데이터는 지금 2015-2019까지밖에 없다. 어쩔수 없이 주식 데이터를 조금 포기해야한다.`

```python
# Now we can merge stock_data , Tweet
merged_data = pd.merge(Tweet,stock_data, how='inner', on=['ticker_symbol', 'date'])

merged_data.head()
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
      <th>tweet_id</th>
      <th>writer</th>
      <th>post_date</th>
      <th>body</th>
      <th>comment_num</th>
      <th>retweet_num</th>
      <th>like_num</th>
      <th>date</th>
      <th>ticker_symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550804137118801921</td>
      <td>DozenStocks</td>
      <td>2015-01-02 00:01:54</td>
      <td>S&P100 #Stocks Performance $HD $LOW $SBUX $TGT...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-02</td>
      <td>AMZN</td>
      <td>312.580</td>
      <td>314.750</td>
      <td>306.960</td>
      <td>308.520</td>
      <td>308.520</td>
      <td>2783200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>550806376092807168</td>
      <td>craigbuj</td>
      <td>2015-01-02 00:10:48</td>
      <td>perfectly trading the S&P 500 in 2014 $FB $MU ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-02</td>
      <td>AMZN</td>
      <td>312.580</td>
      <td>314.750</td>
      <td>306.960</td>
      <td>308.520</td>
      <td>308.520</td>
      <td>2783200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>550808188237279232</td>
      <td>1Copenut</td>
      <td>2015-01-02 00:18:00</td>
      <td>@KyleRohde @thehilker It could be, but that's ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-02</td>
      <td>AMZN</td>
      <td>312.580</td>
      <td>314.750</td>
      <td>306.960</td>
      <td>308.520</td>
      <td>308.520</td>
      <td>2783200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>550810934902390785</td>
      <td>DaveHegedus</td>
      <td>2015-01-02 00:28:55</td>
      <td>RBC Capitals Top Internet Surprise Predictions...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-02</td>
      <td>AMZN</td>
      <td>312.580</td>
      <td>314.750</td>
      <td>306.960</td>
      <td>308.520</td>
      <td>308.520</td>
      <td>2783200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>550821049059655681</td>
      <td>heronjohnson2</td>
      <td>2015-01-02 01:09:06</td>
      <td>FedEx, UPS Recalculate, Raise Rates  http://ao...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-02</td>
      <td>AMZN</td>
      <td>312.580</td>
      <td>314.750</td>
      <td>306.960</td>
      <td>308.520</td>
      <td>308.520</td>
      <td>2783200</td>
    </tr>
  </tbody>
</table>
</div>

`Merging the tweet data with the stock data allows for a combined analysis of how social media activity (tweets) might correlate with stock price movements. `

```python
# Extract the last 5 comments for each company on the last date
merged_data[merged_data['date'] == merged_data['date'].max()].groupby('ticker_symbol').head(5)[['ticker_symbol', 'body']]

# 댓글에 광고있는거같은데 저런건 지우는게 맞을까? 
# 아님 저것도 그때의 추세에 따라서 나오는 광고이니까 그 추세를 반영하는 데이터일까?
# Don't miss our next FREE OPTION TRADE.  Sign up for our Daily Free Trades at http://ow.ly/OIWZ30q4T8t $NVDA $TSLA $GS $C $WFC $GOOGL $FB
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
      <th>ticker_symbol</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3702720</th>
      <td>TSLA</td>
      <td>$TSLA a gap fill is due at 395.22 which also a...</td>
    </tr>
    <tr>
      <th>3702721</th>
      <td>TSLA</td>
      <td>Don't miss our next FREE OPTION TRADE.  Sign u...</td>
    </tr>
    <tr>
      <th>3702722</th>
      <td>TSLA</td>
      <td>"Monet told investigators he regularly uses hi...</td>
    </tr>
    <tr>
      <th>3702723</th>
      <td>TSLA</td>
      <td>These assets are seeing a jump in tweets $ETH ...</td>
    </tr>
    <tr>
      <th>3702724</th>
      <td>TSLA</td>
      <td>#ProfessionalVictim Elon Musk,  lacks self-awa...</td>
    </tr>
    <tr>
      <th>3703777</th>
      <td>AAPL</td>
      <td>Apple & Facebook Had Best Year Out of All FAAN...</td>
    </tr>
    <tr>
      <th>3703778</th>
      <td>AAPL</td>
      <td>Why 2020 Could Be Another Big Year for FAANG S...</td>
    </tr>
    <tr>
      <th>3703779</th>
      <td>AAPL</td>
      <td>$AAPL is the star of the option plans still wi...</td>
    </tr>
    <tr>
      <th>3703780</th>
      <td>AAPL</td>
      <td>Do yourself a favor and watch this video and #...</td>
    </tr>
    <tr>
      <th>3703781</th>
      <td>AAPL</td>
      <td>$SPY $AAPL $QQQ $TVIX  RIP Futures hanging on ...</td>
    </tr>
    <tr>
      <th>3704460</th>
      <td>GOOGL</td>
      <td>Don't miss our next FREE OPTION TRADE.  Sign u...</td>
    </tr>
    <tr>
      <th>3704461</th>
      <td>GOOGL</td>
      <td>With past performance like this, how can you n...</td>
    </tr>
    <tr>
      <th>3704462</th>
      <td>GOOGL</td>
      <td>$IBM $GOOGL</td>
    </tr>
    <tr>
      <th>3704463</th>
      <td>GOOGL</td>
      <td>The ribeye chilly steak burrito didn’t suck ev...</td>
    </tr>
    <tr>
      <th>3704464</th>
      <td>GOOGL</td>
      <td>$GOOGL #fintech  #GooglePay2020stamps</td>
    </tr>
    <tr>
      <th>3704580</th>
      <td>GOOG</td>
      <td>Why 2020 Could Be Another Big Year for FAANG S...</td>
    </tr>
    <tr>
      <th>3704581</th>
      <td>GOOG</td>
      <td>FANGMAN group of leading stocks had the larges...</td>
    </tr>
    <tr>
      <th>3704582</th>
      <td>GOOG</td>
      <td>The $FIT / $GOOG deal will only fail if the DO...</td>
    </tr>
    <tr>
      <th>3704583</th>
      <td>GOOG</td>
      <td>$AAPL $GOOG $MSFT $FB $AMZN $ACB my watchlisth...</td>
    </tr>
    <tr>
      <th>3704584</th>
      <td>GOOG</td>
      <td>$GOOGAnd there it is</td>
    </tr>
    <tr>
      <th>3704694</th>
      <td>MSFT</td>
      <td>Why 2020 Could Be Another Big Year for FAANG S...</td>
    </tr>
    <tr>
      <th>3704695</th>
      <td>MSFT</td>
      <td>FANGMAN group of leading stocks had the larges...</td>
    </tr>
    <tr>
      <th>3704696</th>
      <td>MSFT</td>
      <td>1/20 220 calls or some such?  I'm still studyi...</td>
    </tr>
    <tr>
      <th>3704697</th>
      <td>MSFT</td>
      <td>GREAT NEWS!!! Get in while you can! https://yo...</td>
    </tr>
    <tr>
      <th>3704698</th>
      <td>MSFT</td>
      <td>$MSFT Microsoft says North Korea-linked hacker...</td>
    </tr>
    <tr>
      <th>3704912</th>
      <td>AMZN</td>
      <td>$AMZN today calls traded twice as much as puts...</td>
    </tr>
    <tr>
      <th>3704913</th>
      <td>AMZN</td>
      <td>$TSLA Why did $TSLA drop today?Any real data r...</td>
    </tr>
    <tr>
      <th>3704914</th>
      <td>AMZN</td>
      <td>FANGMAN group of leading stocks had the larges...</td>
    </tr>
    <tr>
      <th>3704915</th>
      <td>AMZN</td>
      <td>Looks like you are still long on $AMZN :)</td>
    </tr>
    <tr>
      <th>3704916</th>
      <td>AMZN</td>
      <td>Lets pull bank statements with these 909 polit...</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Descriptive statistics
merged_data.describe()
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
      <th>tweet_id</th>
      <th>post_date</th>
      <th>comment_num</th>
      <th>retweet_num</th>
      <th>like_num</th>
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3705386.000</td>
      <td>3705386</td>
      <td>3705386.000</td>
      <td>3705386.000</td>
      <td>3705386.000</td>
      <td>3705386</td>
      <td>3705386.000</td>
      <td>3705386.000</td>
      <td>3705386.000</td>
      <td>3705386.000</td>
      <td>3705386.000</td>
      <td>3705386.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>882525760528652032.000</td>
      <td>2017-07-05 09:05:12.002687744</td>
      <td>0.287</td>
      <td>0.619</td>
      <td>2.032</td>
      <td>2017-07-04 18:25:45.746003200</td>
      <td>460.240</td>
      <td>465.335</td>
      <td>454.759</td>
      <td>460.299</td>
      <td>458.064</td>
      <td>20483975.392</td>
    </tr>
    <tr>
      <th>min</th>
      <td>550804137118801920.000</td>
      <td>2015-01-02 00:01:54</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2015-01-02 00:00:00</td>
      <td>40.340</td>
      <td>40.740</td>
      <td>39.720</td>
      <td>40.290</td>
      <td>36.070</td>
      <td>347500.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>719627319057368064.000</td>
      <td>2016-04-11 20:44:57.249999872</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2016-04-11 00:00:00</td>
      <td>128.620</td>
      <td>129.620</td>
      <td>127.850</td>
      <td>128.700</td>
      <td>119.045</td>
      <td>3790200.000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>879355647308972032.000</td>
      <td>2017-06-26 15:08:18</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2017-06-26 00:00:00</td>
      <td>255.250</td>
      <td>259.000</td>
      <td>250.200</td>
      <td>255.030</td>
      <td>255.030</td>
      <td>10517700.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1054196452016192512.000</td>
      <td>2018-10-22 02:23:14.249999872</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>2018-10-22 00:00:00</td>
      <td>703.670</td>
      <td>712.110</td>
      <td>698.000</td>
      <td>705.060</td>
      <td>705.060</td>
      <td>30808700.000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1212160477159206912.000</td>
      <td>2019-12-31 23:55:53</td>
      <td>631.000</td>
      <td>999.000</td>
      <td>999.000</td>
      <td>2019-12-31 00:00:00</td>
      <td>2038.110</td>
      <td>2050.500</td>
      <td>2013.000</td>
      <td>2039.510</td>
      <td>2039.510</td>
      <td>169164000.000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>192552336193193664.000</td>
      <td>NaN</td>
      <td>1.856</td>
      <td>7.032</td>
      <td>13.392</td>
      <td>NaN</td>
      <td>465.909</td>
      <td>470.255</td>
      <td>460.932</td>
      <td>465.874</td>
      <td>467.495</td>
      <td>22829121.626</td>
    </tr>
  </tbody>
</table>
</div>

# 4. EDA

```python
# Plot time series of stock prices for each company
plt.figure(figsize=(14, 7))
sns.lineplot(data=merged_data, x='date', y='close', hue='ticker_symbol')
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(title='Ticker Symbol')
plt.show()
```

---

KeyboardInterrupt                         Traceback (most recent call last)

Cell In[439], line 3
1 # Plot time series of stock prices for each company
2 plt.figure(figsize=(14, 7))
----> 3 sns.lineplot(data=merged_data, x='date', y='close', hue='ticker_symbol')
4 plt.title('Stock Prices Over Time')
5 plt.xlabel('Date')

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/seaborn/relational.py:500, in lineplot(data, x, y, hue, size, style, units, palette, hue_order, hue_norm, sizes, size_order, size_norm, dashes, markers, style_order, estimator, errorbar, n_boot, seed, orient, sort, err_style, err_kws, legend, ci, ax, **kwargs)
497 if not p.has_xy_data:
498     return ax
--> 500 p._attach(ax)
502 # Other functions have color as an explicit param,
503 # and we should probably do that here too
504 color = kwargs.pop("color", kwargs.pop("c", None))

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/seaborn/_base.py:1127, in VectorPlotter._attach(self, obj, allowed_types, log_scale)
1125 # Now actually update the matplotlib objects to do the conversion we want
1126 grouped = self.plot_data[var].groupby(self.converters[var], sort=False)
-> 1127 for converter, seed_data in grouped:
1128     if self.var_types[var] == "categorical":
1129         if self._var_ordered[var]:

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/groupby/ops.py:602, in BaseGrouper.get_iterator(self, data, axis)
591 def get_iterator(
592     self, data: NDFrameT, axis: AxisInt = 0
593 ) -> Iterator[tuple[Hashable, NDFrameT]]:
594     """
595     Groupby iterator
596
(...)
600     for each group
601     """
--> 602     splitter = self._get_splitter(data, axis=axis)
603     keys = self.group_keys_seq
604     yield from zip(keys, splitter)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/groupby/ops.py:613, in BaseGrouper._get_splitter(self, data, axis)
606 @final
607 def _get_splitter(self, data: NDFrame, axis: AxisInt = 0) -> DataSplitter:
608     """
609     Returns
610     -------
611     Generator yielding subsetted objects
612     """
--> 613     ids, _, ngroups = self.group_info
614     return _get_splitter(
615         data,
616         ids,
(...)
620         axis=axis,
621     )

File properties.pyx:36, in pandas._libs.properties.CachedProperty.__get__()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/groupby/ops.py:729, in BaseGrouper.group_info(self)
727 @cache_readonly
728 def group_info(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
--> 729     comp_ids, obs_group_ids = self._get_compressed_codes()
731     ngroups = len(obs_group_ids)
732     comp_ids = ensure_platform_int(comp_ids)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/groupby/ops.py:753, in BaseGrouper._get_compressed_codes(self)
750     # FIXME: compress_group_index's second return value is int64, not intp
752 ping = self.groupings[0]
--> 753 return ping.codes, np.arange(len(ping.group_index), dtype=np.intp)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/groupby/grouper.py:691, in Grouping.codes(self)
689 @property
690 def codes(self) -> npt.NDArray[np.signedinteger]:
--> 691     return self._codes_and_uniques[0]

File properties.pyx:36, in pandas._libs.properties.CachedProperty.__get__()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/groupby/grouper.py:801, in Grouping._codes_and_uniques(self)
796     uniques = self._uniques
797 else:
798     # GH35667, replace dropna=False with use_na_sentinel=False
799     # error: Incompatible types in assignment (expression has type "Union[
800     # ndarray[Any, Any], Index]", variable has type "Categorical")
--> 801     codes, uniques = algorithms.factorize(  # type: ignore[assignment]
802         self.grouping_vector, sort=self._sort, use_na_sentinel=self._dropna
803     )
804 return codes, uniques

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/algorithms.py:795, in factorize(values, sort, use_na_sentinel, size_hint)
792             # Don't modify (potentially user-provided) array
793             values = np.where(null_mask, na_value, values)
--> 795     codes, uniques = factorize_array(
796         values,
797         use_na_sentinel=use_na_sentinel,
798         size_hint=size_hint,
799     )
801 if sort and len(uniques) > 0:
802     uniques, codes = safe_sort(
803         uniques,
804         codes,
(...)
807         verify=False,
808     )

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/algorithms.py:595, in factorize_array(values, use_na_sentinel, size_hint, na_value, mask)
592 hash_klass, values = _get_hashtable_algo(values)
594 table = hash_klass(size_hint or len(values))
--> 595 uniques, codes = table.factorize(
596     values,
597     na_sentinel=-1,
598     na_value=na_value,
599     mask=mask,
600     ignore_na=use_na_sentinel,
601 )
603 # re-cast e.g. i8->dt64/td64, uint8->bool
604 uniques = _reconstruct_data(uniques, original.dtype, original)

KeyboardInterrupt:
![png](tweet_stock_files/tweet_stock_25_1.png)

```python
# Group by date and ticker symbol to get the number of tweets
tweet_counts = merged_data.groupby(['date', 'ticker_symbol']).size().unstack(fill_value=0)

# Resample to 60-day intervals
tweet_counts_60d = tweet_counts.resample('60D').sum()

# Ensure the index is sorted
tweet_counts_60d = tweet_counts_60d.sort_index()

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))

# Stacking bars
bottom = np.zeros(len(tweet_counts_60d))
for ticker in tweet_counts_60d.columns:
    ax.bar(tweet_counts_60d.index, tweet_counts_60d[ticker], 
           bottom=bottom, label=ticker, width=60,edgecolor='black')
    bottom += tweet_counts_60d[ticker]

plt.title('Number of Tweets Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.legend(title='Ticker Symbol')
plt.show()
```

![png](tweet_stock_files/tweet_stock_26_0.png)

## Feature Engineering

```python
# Feature Engineering for Stock Price Data
stock_data['ma_7'] = stock_data.groupby('ticker_symbol')['close'].transform(lambda x: x.rolling(window=7).mean())
stock_data['ma_30'] = stock_data.groupby('ticker_symbol')['close'].transform(lambda x: x.rolling(window=30).mean())
stock_data['volatility_7'] = stock_data.groupby('ticker_symbol')['close'].transform(lambda x: x.rolling(window=7).std())

# Ensure no NaN values remain from rolling operations
stock_data.dropna(inplace=True)

# Feature Engineering for Tweet Data
Tweet['total_engagement'] = Tweet['comment_num'] + Tweet['retweet_num'] + Tweet['like_num']

# Calculate daily tweet volume for each ticker symbol
tweet_volume = Tweet.groupby(['date', 'ticker_symbol']).size().reset_index(name='tweet_volume')

# Merge tweet volume and engagements back into the Tweet dataframe
Tweet = Tweet.merge(tweet_volume, on=['date', 'ticker_symbol'], how='left')

# Aggregating tweet features by date and ticker_symbol
tweet_features = Tweet.groupby(['date', 'ticker_symbol']).agg({
    'total_engagement': 'sum',
    'tweet_volume': 'sum'
}).reset_index()

# Merge tweet features into the stock data
stock_data = pd.merge(stock_data, tweet_features, on=['date', 'ticker_symbol'])

# Fill any NaN values in the tweet features with 0 (indicating no tweets)
stock_data[['total_engagement', 'tweet_volume']].fillna(0, inplace=True)

# Display the first few rows to verify
print("Stock Data with Tweet Features:")
stock_data.head()
```

Stock Data with Tweet Features:

/var/folders/12/vb6sch5j7lx93jxvhtgjvhyh0000gn/T/ipykernel_61982/1428808705.py:28: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
stock_data[['total_engagement', 'tweet_volume']].fillna(0, inplace=True)

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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
      <th>ticker_symbol</th>
      <th>ma_7</th>
      <th>ma_30</th>
      <th>volatility_7</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>289.930</td>
      <td>293.680</td>
      <td>289.520</td>
      <td>293.650</td>
      <td>292.955</td>
      <td>25201400</td>
      <td>AAPL</td>
      <td>300.349</td>
      <td>313.548</td>
      <td>5.029</td>
      <td>2288</td>
      <td>466489</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-12-30</td>
      <td>289.460</td>
      <td>292.690</td>
      <td>285.220</td>
      <td>291.520</td>
      <td>290.830</td>
      <td>36028600</td>
      <td>AAPL</td>
      <td>297.761</td>
      <td>312.359</td>
      <td>4.014</td>
      <td>3256</td>
      <td>535824</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-12-27</td>
      <td>291.120</td>
      <td>293.970</td>
      <td>288.120</td>
      <td>289.800</td>
      <td>289.114</td>
      <td>36566500</td>
      <td>AAPL</td>
      <td>295.849</td>
      <td>311.365</td>
      <td>4.183</td>
      <td>1194</td>
      <td>145161</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-26</td>
      <td>284.820</td>
      <td>289.980</td>
      <td>284.700</td>
      <td>289.910</td>
      <td>289.224</td>
      <td>23280300</td>
      <td>AAPL</td>
      <td>294.637</td>
      <td>310.311</td>
      <td>4.537</td>
      <td>2613</td>
      <td>477481</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-24</td>
      <td>284.690</td>
      <td>284.890</td>
      <td>282.920</td>
      <td>284.270</td>
      <td>283.597</td>
      <td>12119700</td>
      <td>AAPL</td>
      <td>292.419</td>
      <td>309.119</td>
      <td>5.321</td>
      <td>1348</td>
      <td>123904</td>
    </tr>
  </tbody>
</table>
</div>

```python
print("Tweet Data with Features:")
Tweet.head()
```

Tweet Data with Features:

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
      <th>tweet_id</th>
      <th>writer</th>
      <th>post_date</th>
      <th>body</th>
      <th>comment_num</th>
      <th>retweet_num</th>
      <th>like_num</th>
      <th>date</th>
      <th>ticker_symbol</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550441509175443456</td>
      <td>VisualStockRSRC</td>
      <td>2015-01-01 00:00:57</td>
      <td>lx21 made $10,008  on $AAPL -Check it out! htt...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>1</td>
      <td>299</td>
    </tr>
    <tr>
      <th>1</th>
      <td>550441672312512512</td>
      <td>KeralaGuy77</td>
      <td>2015-01-01 00:01:36</td>
      <td>Insanity of today weirdo massive selling. $aap...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>0</td>
      <td>299</td>
    </tr>
    <tr>
      <th>2</th>
      <td>550441732014223360</td>
      <td>DozenStocks</td>
      <td>2015-01-01 00:01:50</td>
      <td>S&P100 #Stocks Performance $HD $LOW $SBUX $TGT...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-01-01</td>
      <td>AMZN</td>
      <td>0</td>
      <td>131</td>
    </tr>
    <tr>
      <th>3</th>
      <td>550442977802207232</td>
      <td>ShowDreamCar</td>
      <td>2015-01-01 00:06:47</td>
      <td>$GM $TSLA: Volkswagen Pushes 2014 Record Recal...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>TSLA</td>
      <td>1</td>
      <td>99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>550443807834402816</td>
      <td>i_Know_First</td>
      <td>2015-01-01 00:10:05</td>
      <td>Swing Trading: Up To 8.91% Return In 14 Days h...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>1</td>
      <td>299</td>
    </tr>
  </tbody>
</table>
</div>

# 6. Sentimelment Analysis (NLP)

1. Text Cleaning
2. POS 주어 동사
3. Tokenizer + Lemmatizer
4. Sentiment
   1. AutoTokenizer
   2. TFAutoModelForSequenceClassification
5. 날짜별로 sum, count, mean
6. 결과 merge to stock_df

```python
stock_data = pd.read_csv('stock_data.csv',index_col=0)
stock_data
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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
      <th>ticker_symbol</th>
      <th>ma_7</th>
      <th>ma_30</th>
      <th>volatility_7</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>976</th>
      <td>2019-12-31</td>
      <td>289.930</td>
      <td>293.680</td>
      <td>289.520</td>
      <td>293.650</td>
      <td>292.955</td>
      <td>25201400</td>
      <td>AAPL</td>
      <td>300.349</td>
      <td>313.548</td>
      <td>5.029</td>
      <td>2288.000</td>
      <td>466489.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>977</th>
      <td>2019-12-30</td>
      <td>289.460</td>
      <td>292.690</td>
      <td>285.220</td>
      <td>291.520</td>
      <td>290.830</td>
      <td>36028600</td>
      <td>AAPL</td>
      <td>297.761</td>
      <td>312.359</td>
      <td>4.014</td>
      <td>3256.000</td>
      <td>535824.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>978</th>
      <td>2019-12-27</td>
      <td>291.120</td>
      <td>293.970</td>
      <td>288.120</td>
      <td>289.800</td>
      <td>289.114</td>
      <td>36566500</td>
      <td>AAPL</td>
      <td>295.849</td>
      <td>311.365</td>
      <td>4.183</td>
      <td>1194.000</td>
      <td>145161.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>979</th>
      <td>2019-12-26</td>
      <td>284.820</td>
      <td>289.980</td>
      <td>284.700</td>
      <td>289.910</td>
      <td>289.224</td>
      <td>23280300</td>
      <td>AAPL</td>
      <td>294.637</td>
      <td>310.311</td>
      <td>4.537</td>
      <td>2613.000</td>
      <td>477481.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>980</th>
      <td>2019-12-24</td>
      <td>284.690</td>
      <td>284.890</td>
      <td>282.920</td>
      <td>284.270</td>
      <td>283.597</td>
      <td>12119700</td>
      <td>AAPL</td>
      <td>292.419</td>
      <td>309.119</td>
      <td>5.321</td>
      <td>1348.000</td>
      <td>123904.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>37968</th>
      <td>2015-01-08</td>
      <td>212.810</td>
      <td>213.800</td>
      <td>210.010</td>
      <td>210.620</td>
      <td>210.620</td>
      <td>3442500</td>
      <td>TSLA</td>
      <td>200.196</td>
      <td>206.351</td>
      <td>7.609</td>
      <td>281.000</td>
      <td>56644.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37969</th>
      <td>2015-01-07</td>
      <td>213.350</td>
      <td>214.780</td>
      <td>209.780</td>
      <td>210.950</td>
      <td>210.950</td>
      <td>2968400</td>
      <td>TSLA</td>
      <td>202.750</td>
      <td>206.146</td>
      <td>7.816</td>
      <td>283.000</td>
      <td>63504.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37970</th>
      <td>2015-01-06</td>
      <td>210.060</td>
      <td>214.200</td>
      <td>204.210</td>
      <td>211.280</td>
      <td>211.280</td>
      <td>6261900</td>
      <td>TSLA</td>
      <td>205.523</td>
      <td>206.131</td>
      <td>6.673</td>
      <td>216.000</td>
      <td>57600.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37971</th>
      <td>2015-01-05</td>
      <td>214.550</td>
      <td>216.500</td>
      <td>207.160</td>
      <td>210.090</td>
      <td>210.090</td>
      <td>5368500</td>
      <td>TSLA</td>
      <td>208.009</td>
      <td>206.319</td>
      <td>3.653</td>
      <td>309.000</td>
      <td>77284.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37972</th>
      <td>2015-01-02</td>
      <td>222.870</td>
      <td>223.250</td>
      <td>213.260</td>
      <td>219.310</td>
      <td>219.310</td>
      <td>4764400</td>
      <td>TSLA</td>
      <td>210.160</td>
      <td>206.818</td>
      <td>5.184</td>
      <td>218.000</td>
      <td>40401.000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>7544 rows × 14 columns</p>
</div>

### 6.1. Text Cleaning (Tokenizer + Lemmatizer)

```python
def get_wordnet_pos(word):
    treebank_tag = pos_tag([word])[0][1][0].upper()
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
```

```python
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'$\w+', '', text)  # Remove ticker tags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation 구두점 제거
  
    # Initialization the twitter tokenizer
    tk = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True) 
    # Initialization the lemmatizer
    lemmatizer = WordNetLemmatizer()  
    # Trying to avoid deleting the negative verbs as it affects the meaning of the tweets.
    stop_words = stopwords.words('english') + ["i'll","i'm", "should", "could"]
    negative_verbs = [ "shan't",'shouldn',"shouldn't",'wasn','weren','won','wouldn',
                      'aren','couldn','didn','doesn','hadn','hasn','haven','isn',
                      'ma','mightn','mustn',"mustn't",'needn',"needn't","wouldn't",
                      "won't","weren't","wasn't","couldn","not","nor","no","mightn't",
                      "isn't","haven't","hadn't","hasn't","didn't","doesn't","aren't",
                      "don't","couldn't","never"]
    stop_words =[word for word in stop_words if word not in negative_verbs ] 
  
    # Tokenize the tweets by twitter tokenzier.
    text = tk.tokenize(text)
    # Choosing the words that don't exist in stopwords, thier lengths are more than 2 letters and then lemmatize them.
    text = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text if word not in stop_words and word not in string.punctuation and len(word)>2 and "." not in word]
    # return the tokens in one sentence 
    text = " ".join(text)
    return text

tqdm.pandas()
Tweet['cleaned_body'] = Tweet['body'].progress_apply(clean_text)
Tweet.to_csv('Tweet_after_cleaned.csv')
Tweet[['body', 'cleaned_body']].head()
```

100%|██████████| 4336445/4336445 [1:10:22<00:00, 1026.93it/s]

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
      <th>body</th>
      <th>cleaned_body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lx21 made $10,008  on $AAPL -Check it out! htt...</td>
      <td>make aapl check learn exe watt imrs cach gmo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Insanity of today weirdo massive selling. $aap...</td>
      <td>insanity today weirdo massive sell aapl bid ce...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S&P100 #Stocks Performance $HD $LOW $SBUX $TGT...</td>
      <td>performance low sbux tgt dvn ibm amzn apa hal ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>$GM $TSLA: Volkswagen Pushes 2014 Record Recal...</td>
      <td>tsla volkswagen push record recall tally high</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Swing Trading: Up To 8.91% Return In 14 Days h...</td>
      <td>swing trading return day mww aapl tsla</td>
    </tr>
  </tbody>
</table>
</div>

```python
Tweet = pd.read_csv('Tweet_after_cleaned.csv',index_col=0)
Tweet.head()
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
      <th>tweet_id</th>
      <th>writer</th>
      <th>post_date</th>
      <th>body</th>
      <th>comment_num</th>
      <th>retweet_num</th>
      <th>like_num</th>
      <th>date</th>
      <th>ticker_symbol</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>cleaned_body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550441509175443456</td>
      <td>VisualStockRSRC</td>
      <td>2015-01-01 00:00:57</td>
      <td>lx21 made $10,008  on $AAPL -Check it out! htt...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>1</td>
      <td>299</td>
      <td>make aapl check learn exe watt imrs cach gmo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>550441672312512512</td>
      <td>KeralaGuy77</td>
      <td>2015-01-01 00:01:36</td>
      <td>Insanity of today weirdo massive selling. $aap...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>0</td>
      <td>299</td>
      <td>insanity today weirdo massive sell aapl bid ce...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>550441732014223360</td>
      <td>DozenStocks</td>
      <td>2015-01-01 00:01:50</td>
      <td>S&P100 #Stocks Performance $HD $LOW $SBUX $TGT...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-01-01</td>
      <td>AMZN</td>
      <td>0</td>
      <td>131</td>
      <td>performance low sbux tgt dvn ibm amzn apa hal ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>550442977802207232</td>
      <td>ShowDreamCar</td>
      <td>2015-01-01 00:06:47</td>
      <td>$GM $TSLA: Volkswagen Pushes 2014 Record Recal...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>TSLA</td>
      <td>1</td>
      <td>99</td>
      <td>tsla volkswagen push record recall tally high</td>
    </tr>
    <tr>
      <th>4</th>
      <td>550443807834402816</td>
      <td>i_Know_First</td>
      <td>2015-01-01 00:10:05</td>
      <td>Swing Trading: Up To 8.91% Return In 14 Days h...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>1</td>
      <td>299</td>
      <td>swing trading return day mww aapl tsla</td>
    </tr>
  </tbody>
</table>
</div>

### 6.2. Sentiment Analysis

#### 6.2.1. bert-base-multilingual-uncased-sentiment

```python
# # Load pre-trained model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
# model = TFAutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# def get_sentiment(text):
#     inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
#     outputs = model(inputs)
#     scores = outputs[0][0].numpy()
#     sentiment = tf.nn.softmax(scores).numpy()
#     return sentiment

# tqdm.pandas()
# Tweet['sentiment'] = Tweet['lemmatized_body'].progress_apply(lambda x: get_sentiment(x).tolist())
# Tweet[['lemmatized_body', 'sentiment']].head()
```

- POS 과정에서 NaN이 생겻다.
- 그래서 fillna(' ') 공백으로 변경해서 해결

#### 6.2.2. vaderSentiment

```python
# !pip install vaderSentiment
```

```python
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from copy import deepcopy

# VADER 분석기 초기화
analyzer = SentimentIntensityAnalyzer()

# 감성 분석 함수
def analyze_sentiment_vader(text):
    scores = analyzer.polarity_scores(text)
    return scores

# 감성 분석 수행
Tweet['cleaned_body'].fillna('', inplace=True)
tqdm.pandas()
sentiment_vader = Tweet['cleaned_body'].progress_apply(lambda x: analyze_sentiment_vader(x))

# VADER 결과를 데이터프레임으로 변환하고 병합
Tweet_vader = deepcopy(Tweet)
sentiment_df = pd.json_normalize(sentiment_vader)
Tweet_vader = pd.concat([Tweet_vader, sentiment_df], axis=1)

# 결과를 CSV 파일로 저장
Tweet_vader.to_csv('Tweet_vader.csv')
Tweet_vader.head()
```

14%|█▎        | 591772/4336445 [00:12<01:20, 46588.32it/s]

---

KeyboardInterrupt                         Traceback (most recent call last)

Cell In[34], line 17
15 Tweet['cleaned_body'].fillna('', inplace=True)
16 tqdm.pandas()
---> 17 sentiment_vader = Tweet['cleaned_body'].progress_apply(lambda x: analyze_sentiment_vader(x))
19 # VADER 결과를 데이터프레임으로 변환하고 병합
20 Tweet_vader = deepcopy(Tweet)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/tqdm/std.py:920, in tqdm.pandas.<locals>.inner_generator.<locals>.inner(df, func, *args, **kwargs)
917 # Apply the provided function (in **kwargs)
918 # on the df using our wrapper (which provides bar updating)
919 try:
--> 920     return getattr(df, df_function)(wrapper, **kwargs)
921 finally:
922     t.close()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/series.py:4753, in Series.apply(self, func, convert_dtype, args, by_row, **kwargs)
4625 def apply(
4626     self,
4627     func: AggFuncType,
(...)
4632     **kwargs,
4633 ) -> DataFrame | Series:
4634     """
4635     Invoke function on values of Series.
4636
(...)
4751     dtype: float64
4752     """
-> 4753     return SeriesApply(
4754         self,
4755         func,
4756         convert_dtype=convert_dtype,
4757         by_row=by_row,
4758         args=args,
4759         kwargs=kwargs,
4760     ).apply()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/apply.py:1207, in SeriesApply.apply(self)
1204     return self.apply_compat()
1206 # self.func is Callable
-> 1207 return self.apply_standard()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/apply.py:1287, in SeriesApply.apply_standard(self)
1281 # row-wise access
1282 # apply doesn't have a `na_action` keyword and for backward compat reasons
1283 # we need to give `na_action="ignore"` for categorical data.
1284 # TODO: remove the `na_action="ignore"` when that default has been changed in
1285 #  Categorical (GH51645).
1286 action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1287 mapped = obj._map_values(
1288     mapper=curried, na_action=action, convert=self.convert_dtype
1289 )
1291 if len(mapped) and isinstance(mapped[0], ABCSeries):
1292     # GH#43986 Need to do list(mapped) in order to get treated as nested
1293     #  See also GH#25959 regarding EA support
1294     return obj._constructor_expanddim(list(mapped), index=obj.index)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/base.py:921, in IndexOpsMixin._map_values(self, mapper, na_action, convert)
918 if isinstance(arr, ExtensionArray):
919     return arr.map(mapper, na_action=na_action)
--> 921 return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/algorithms.py:1814, in map_array(arr, mapper, na_action, convert)
1812 values = arr.astype(object, copy=False)
1813 if na_action is None:
-> 1814     return lib.map_infer(values, mapper, convert=convert)
1815 else:
1816     return lib.map_infer_mask(
1817         values, mapper, mask=isna(values).view(np.uint8), convert=convert
1818     )

File lib.pyx:2920, in pandas._libs.lib.map_infer()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/tqdm/std.py:915, in tqdm.pandas.<locals>.inner_generator.<locals>.inner.<locals>.wrapper(*args, **kwargs)
909 def wrapper(*args, **kwargs):
910     # update tbar correctly
911     # it seems `pandas apply` calls `func` twice
912     # on the first column/row to decide whether it can
913     # take a fast or slow code path; so stop when t.total==t.n
914     t.update(n=1 if not t.total or t.n < t.total else 0)
--> 915     return func(*args, **kwargs)

Cell In[34], line 17, in <lambda>(x)
15 Tweet['cleaned_body'].fillna('', inplace=True)
16 tqdm.pandas()
---> 17 sentiment_vader = Tweet['cleaned_body'].progress_apply(lambda x: analyze_sentiment_vader(x))
19 # VADER 결과를 데이터프레임으로 변환하고 병합
20 Tweet_vader = deepcopy(Tweet)

Cell In[34], line 11, in analyze_sentiment_vader(text)
10 def analyze_sentiment_vader(text):
---> 11     scores = analyzer.polarity_scores(text)
12     return scores

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/vaderSentiment/vaderSentiment.py:242, in SentimentIntensityAnalyzer.polarity_scores(self, text)
240 prev_space = True
241 for chr in text:
--> 242     if chr in self.emojis:
243         # get the textual description
244         description = self.emojis[chr]
245         if not prev_space:

KeyboardInterrupt:

```python
senti_vader = sentiment_df
senti_vader.to_csv('senti_vader.csv',index=False)
```

---

NameError                                 Traceback (most recent call last)

Cell In[30], line 1
----> 1 senti_vader = sentiment_df
2 senti_vader.to_csv('senti_vader.csv',index=False)

NameError: name 'sentiment_df' is not defined

```python
Tweet_vader = pd.read_csv('Tweet_vader.csv',index_col=False)
# engagement_weight 계산 - 1을 더해서 가중치가 1부터 시작하게 함
Tweet_vader['engagement_weight'] = Tweet_vader['total_engagement'] + 1

# compound에 가중치를 적용한 새로운 sentiment score 계산
Tweet_vader['weighted_compound'] = Tweet_vader['compound'] * Tweet_vader['engagement_weight']

# weighted_compound의 최종 값이 0인 경우, 원래 compound 값을 유지
Tweet_vader['final_compound'] = Tweet_vader.apply(lambda x: x['compound'] if x['total_engagement'] == 0 else x['weighted_compound'], axis=1)

# 결과 출력
senti_vader_mean = Tweet_vader.groupby(['date', 'ticker_symbol'])['final_compound'].mean().reset_index()
senti_vader_mean.columns = ['date', 'ticker_symbol','polarity']
senti_vader_mean
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
      <th>date</th>
      <th>ticker_symbol</th>
      <th>polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>1.580</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01</td>
      <td>AMZN</td>
      <td>-1.474</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01</td>
      <td>GOOG</td>
      <td>0.456</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01</td>
      <td>GOOGL</td>
      <td>0.329</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01</td>
      <td>MSFT</td>
      <td>0.425</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10943</th>
      <td>2019-12-31</td>
      <td>AMZN</td>
      <td>0.724</td>
    </tr>
    <tr>
      <th>10944</th>
      <td>2019-12-31</td>
      <td>GOOG</td>
      <td>0.295</td>
    </tr>
    <tr>
      <th>10945</th>
      <td>2019-12-31</td>
      <td>GOOGL</td>
      <td>0.382</td>
    </tr>
    <tr>
      <th>10946</th>
      <td>2019-12-31</td>
      <td>MSFT</td>
      <td>2.049</td>
    </tr>
    <tr>
      <th>10947</th>
      <td>2019-12-31</td>
      <td>TSLA</td>
      <td>2.360</td>
    </tr>
  </tbody>
</table>
<p>10948 rows × 3 columns</p>
</div>

#### 6.2.3. textblob

```python
# !pip install textblob
```

Collecting textblob
Downloading textblob-0.18.0.post0-py3-none-any.whl.metadata (4.5 kB)
Requirement already satisfied: nltk>=3.8 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from textblob) (3.8.1)
Requirement already satisfied: click in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from nltk>=3.8->textblob) (8.1.7)
Requirement already satisfied: joblib in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from nltk>=3.8->textblob) (1.3.2)
Requirement already satisfied: regex>=2021.8.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from nltk>=3.8->textblob) (2023.10.3)
Requirement already satisfied: tqdm in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from nltk>=3.8->textblob) (4.66.1)
Downloading textblob-0.18.0.post0-py3-none-any.whl (626 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m626.3/626.3 kB[0m [31m288.3 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
[?25hInstalling collected packages: textblob
Successfully installed textblob-0.18.0.post0

```python
from textblob import TextBlob

# 감성 분석 함수
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment

# 가중치 기반 폴라리티 계산 함수
def calculate_weighted_polarity(polarity, subjectivity):
    return polarity * (1 - subjectivity)

# 감정 분석 수행
Tweet['cleaned_body'].fillna('', inplace=True)
tqdm.pandas()
sentiment_textblob = Tweet['cleaned_body'].progress_apply(lambda x: analyze_sentiment_textblob(x))

# 감정 점수를 데이터프레임에 추가
Tweet_textblob = deepcopy(Tweet)
Tweet_textblob['polarity'] = sentiment_textblob.apply(lambda x: x.polarity)
Tweet_textblob['subjectivity'] = sentiment_textblob.apply(lambda x: x.subjectivity)

# weighted_polarity 컬럼 생성
Tweet_textblob['weighted_polarity'] = Tweet_textblob.apply(lambda row: calculate_weighted_polarity(row['polarity'], row['subjectivity']), axis=1)

# 결과를 CSV 파일로 저장
Tweet_textblob.to_csv('Tweet_textblob.csv')

# 결과 표시
Tweet_textblob.head()
```

100%|██████████| 4336445/4336445 [13:09<00:00, 5495.27it/s]

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
      <th>tweet_id</th>
      <th>writer</th>
      <th>post_date</th>
      <th>body</th>
      <th>comment_num</th>
      <th>retweet_num</th>
      <th>like_num</th>
      <th>date</th>
      <th>ticker_symbol</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>cleaned_body</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>weighted_polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550441509175443456</td>
      <td>VisualStockRSRC</td>
      <td>2015-01-01 00:00:57</td>
      <td>lx21 made $10,008  on $AAPL -Check it out! htt...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>1</td>
      <td>299</td>
      <td>make aapl check learn exe watt imrs cach gmo</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>550441672312512512</td>
      <td>KeralaGuy77</td>
      <td>2015-01-01 00:01:36</td>
      <td>Insanity of today weirdo massive selling. $aap...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>0</td>
      <td>299</td>
      <td>insanity today weirdo massive sell aapl bid ce...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>550441732014223360</td>
      <td>DozenStocks</td>
      <td>2015-01-01 00:01:50</td>
      <td>S&P100 #Stocks Performance $HD $LOW $SBUX $TGT...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-01-01</td>
      <td>AMZN</td>
      <td>0</td>
      <td>131</td>
      <td>performance low sbux tgt dvn ibm amzn apa hal ...</td>
      <td>0.000</td>
      <td>0.300</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>550442977802207232</td>
      <td>ShowDreamCar</td>
      <td>2015-01-01 00:06:47</td>
      <td>$GM $TSLA: Volkswagen Pushes 2014 Record Recal...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>TSLA</td>
      <td>1</td>
      <td>99</td>
      <td>tsla volkswagen push record recall tally high</td>
      <td>0.160</td>
      <td>0.540</td>
      <td>0.074</td>
    </tr>
    <tr>
      <th>4</th>
      <td>550443807834402816</td>
      <td>i_Know_First</td>
      <td>2015-01-01 00:10:05</td>
      <td>Swing Trading: Up To 8.91% Return In 14 Days h...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>1</td>
      <td>299</td>
      <td>swing trading return day mww aapl tsla</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>

```python
Tweet_textblob = pd.read_csv('Tweet_textblob.csv',index_col=0)
Tweet_textblob['engagement_weight'] = Tweet_textblob['total_engagement'] + 1

# polarity에 가중치를 적용한 새로운 sentiment score 계산
Tweet_textblob['weighted_polarity'] = Tweet_textblob['polarity'] * Tweet_textblob['engagement_weight']

# date와 ticker_symbol 기준으로 weighted_polarity 평균값 계산
senti_textblob_mean = Tweet_textblob.groupby(['date', 'ticker_symbol'])['weighted_polarity'].mean().reset_index()
senti_textblob_mean.columns = ['date', 'ticker_symbol','polarity']
# 결과 출력
senti_textblob_mean
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
      <th>date</th>
      <th>ticker_symbol</th>
      <th>polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>0.735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01</td>
      <td>AMZN</td>
      <td>-0.754</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01</td>
      <td>GOOG</td>
      <td>0.310</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01</td>
      <td>GOOGL</td>
      <td>0.254</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01</td>
      <td>MSFT</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10943</th>
      <td>2019-12-31</td>
      <td>AMZN</td>
      <td>0.358</td>
    </tr>
    <tr>
      <th>10944</th>
      <td>2019-12-31</td>
      <td>GOOG</td>
      <td>0.182</td>
    </tr>
    <tr>
      <th>10945</th>
      <td>2019-12-31</td>
      <td>GOOGL</td>
      <td>0.242</td>
    </tr>
    <tr>
      <th>10946</th>
      <td>2019-12-31</td>
      <td>MSFT</td>
      <td>0.692</td>
    </tr>
    <tr>
      <th>10947</th>
      <td>2019-12-31</td>
      <td>TSLA</td>
      <td>1.231</td>
    </tr>
  </tbody>
</table>
<p>10948 rows × 3 columns</p>
</div>

#### 6.2.4. twitter-xlm-roberta-base-sentiment

```python
# Combine tweets by date and ticker symbol
Tweet.fillna('',inplace=True)
combined_tweets = Tweet.groupby(['date', 'ticker_symbol'])['cleaned_body'].apply(' '.join).reset_index()
combined_tweets['cleaned_body'].fillna('', inplace=True)

# Display the combined tweets
combined_tweets.head(10)
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
      <th>date</th>
      <th>ticker_symbol</th>
      <th>cleaned_body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>make aapl check learn exe watt imrs cach gmo i...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01</td>
      <td>AMZN</td>
      <td>performance low sbux tgt dvn ibm amzn apa hal ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01</td>
      <td>GOOG</td>
      <td>top search aapl baba tsla bac goog intc twtr a...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01</td>
      <td>GOOGL</td>
      <td>year review part end aapl googl spy vxx uso pc...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01</td>
      <td>MSFT</td>
      <td>unp orcl qcom msft aapl top score mega cap rig...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-01-01</td>
      <td>TSLA</td>
      <td>tsla volkswagen push record recall tally high ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015-01-02</td>
      <td>AAPL</td>
      <td>either way youre winnah object brka real money...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015-01-02</td>
      <td>AMZN</td>
      <td>performance low sbux tgt dvn ibm amzn apa hal ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015-01-02</td>
      <td>GOOG</td>
      <td>hand large bugfinders fee goog doesnt google p...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015-01-02</td>
      <td>GOOGL</td>
      <td>note whatever think googl downtrend follow rbc...</td>
    </tr>
  </tbody>
</table>
</div>

```python
from transformers import AutoTokenizer, AutoConfig, TFAutoModelForSequenceClassification
import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm
from copy import deepcopy

# Load pre-trained model and tokenizer
MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# TF
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    output = model(inputs)
    scores = output[0][0].numpy()
    scores = softmax(scores)
  
    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    res = {}
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        # print(f"{i+1}) {l} {np.round(float(s), 4)}")
        res[l] = s
    return dict(sorted(res.items()))

# Apply sentiment analysis with progress bar
tqdm.pandas()
Tweet_auto = deepcopy(combined_tweets)
sentiments = combined_tweets['cleaned_body'].progress_apply(lambda x: pd.Series(get_sentiment(x)))
Tweet_auto = pd.concat([combined_tweets, sentiments], axis=1)
Tweet_auto.to_csv('Tweet_auto.csv')

# Display the results
Tweet_auto.head()
```

---

Exception                                 Traceback (most recent call last)

Cell In[15], line 10
8 # Load pre-trained model and tokenizer
9 MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
---> 10 tokenizer = AutoTokenizer.from_pretrained(MODEL)
11 config = AutoConfig.from_pretrained(MODEL)
13 # TF

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/models/auto/tokenization_auto.py:591, in AutoTokenizer.from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
587     if tokenizer_class is None:
588         raise ValueError(
589             f"Tokenizer class {tokenizer_class_candidate} does not exist or is not currently imported."
590         )
--> 591     return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
593 # Otherwise we have to be creative.
594 # if model is an encoder decoder, the encoder tokenizer class is used by default
595 if isinstance(config, EncoderDecoderConfig):

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/tokenization_utils_base.py:1805, in PreTrainedTokenizerBase.from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs)
1802     else:
1803         logger.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")
-> 1805 return cls._from_pretrained(
1806     resolved_vocab_files,
1807     pretrained_model_name_or_path,
1808     init_configuration,
1809     *init_inputs,
1810     use_auth_token=use_auth_token,
1811     cache_dir=cache_dir,
1812     **kwargs,
1813 )

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/tokenization_utils_base.py:1950, in PreTrainedTokenizerBase._from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, use_auth_token, cache_dir, *init_inputs, **kwargs)
1948 # Instantiate tokenizer.
1949 try:
-> 1950     tokenizer = cls(*init_inputs, **init_kwargs)
1951 except OSError:
1952     raise OSError(
1953         "Unable to load vocabulary from file. "
1954         "Please check that the provided vocabulary is accessible and not corrupted."
1955     )

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/models/xlm_roberta/tokenization_xlm_roberta_fast.py:155, in XLMRobertaTokenizerFast.__init__(self, vocab_file, tokenizer_file, bos_token, eos_token, sep_token, cls_token, unk_token, pad_token, mask_token, **kwargs)
139 def __init__(
140     self,
141     vocab_file=None,
(...)
151 ):
152     # Mask token behave like a normal word, i.e. include the space before it
153     mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
--> 155     super().__init__(
156         vocab_file,
157         tokenizer_file=tokenizer_file,
158         bos_token=bos_token,
159         eos_token=eos_token,
160         sep_token=sep_token,
161         cls_token=cls_token,
162         unk_token=unk_token,
163         pad_token=pad_token,
164         mask_token=mask_token,
165         **kwargs,
166     )
168     self.vocab_file = vocab_file
169     self.can_save_slow_tokenizer = False if not self.vocab_file else True

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/tokenization_utils_fast.py:110, in PreTrainedTokenizerFast.__init__(self, *args, **kwargs)
107     fast_tokenizer = tokenizer_object
108 elif fast_tokenizer_file is not None and not from_slow:
109     # We have a serialization from tokenizers which let us directly build the backend
--> 110     fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
111 elif slow_tokenizer is not None:
112     # We need to convert a slow tokenizer to build the backend
113     fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)

Exception: data did not match any variant of untagged enum PyPreTokenizerTypeWrapper at line 78 column 3

```python
Tweet_auto = pd.read_csv('Tweet_auto.csv',index_col=0)
Tweet_auto.head()
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
      <th>date</th>
      <th>ticker_symbol</th>
      <th>cleaned_body</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>make aapl check learn exe watt imrs cach gmo i...</td>
      <td>0.316</td>
      <td>0.374</td>
      <td>0.310</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01</td>
      <td>AMZN</td>
      <td>performance low sbux tgt dvn ibm amzn apa hal ...</td>
      <td>0.376</td>
      <td>0.395</td>
      <td>0.229</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01</td>
      <td>GOOG</td>
      <td>top search aapl baba tsla bac goog intc twtr a...</td>
      <td>0.282</td>
      <td>0.409</td>
      <td>0.309</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01</td>
      <td>GOOGL</td>
      <td>year review part end aapl googl spy vxx uso pc...</td>
      <td>0.305</td>
      <td>0.387</td>
      <td>0.308</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01</td>
      <td>MSFT</td>
      <td>unp orcl qcom msft aapl top score mega cap rig...</td>
      <td>0.332</td>
      <td>0.393</td>
      <td>0.275</td>
    </tr>
  </tbody>
</table>
</div>

#### 6.2.5. tweetnlp

- https://github.com/cardiffnlp/tweetnlp
- Sentiment Analysis Model : https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

```python
import tweetnlp

model = tweetnlp.load_model('sentiment')
tweetnlp_sentiments = Tweet['cleaned_body'].progress_apply(lambda x: pd.Series(model.sentiment(x, return_probability=True)))
tweetnlp_sentiments.head()
```

Some weights of the model checkpoint at cardiffnlp/twitter-roberta-base-sentiment-latest were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']

- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
  0%|          | 912/4336445 [00:14<18:46:45, 64.13it/s]

---

KeyboardInterrupt                         Traceback (most recent call last)

Cell In[21], line 4
1 import tweetnlp
3 model = tweetnlp.load_model('sentiment')
----> 4 tweetnlp_sentiments = Tweet['cleaned_body'].progress_apply(lambda x: pd.Series(model.sentiment(x, return_probability=True)))
5 tweetnlp_sentiments.head()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/tqdm/std.py:920, in tqdm.pandas.<locals>.inner_generator.<locals>.inner(df, func, *args, **kwargs)
917 # Apply the provided function (in **kwargs)
918 # on the df using our wrapper (which provides bar updating)
919 try:
--> 920     return getattr(df, df_function)(wrapper, **kwargs)
921 finally:
922     t.close()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/series.py:4753, in Series.apply(self, func, convert_dtype, args, by_row, **kwargs)
4625 def apply(
4626     self,
4627     func: AggFuncType,
(...)
4632     **kwargs,
4633 ) -> DataFrame | Series:
4634     """
4635     Invoke function on values of Series.
4636
(...)
4751     dtype: float64
4752     """
-> 4753     return SeriesApply(
4754         self,
4755         func,
4756         convert_dtype=convert_dtype,
4757         by_row=by_row,
4758         args=args,
4759         kwargs=kwargs,
4760     ).apply()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/apply.py:1207, in SeriesApply.apply(self)
1204     return self.apply_compat()
1206 # self.func is Callable
-> 1207 return self.apply_standard()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/apply.py:1287, in SeriesApply.apply_standard(self)
1281 # row-wise access
1282 # apply doesn't have a `na_action` keyword and for backward compat reasons
1283 # we need to give `na_action="ignore"` for categorical data.
1284 # TODO: remove the `na_action="ignore"` when that default has been changed in
1285 #  Categorical (GH51645).
1286 action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1287 mapped = obj._map_values(
1288     mapper=curried, na_action=action, convert=self.convert_dtype
1289 )
1291 if len(mapped) and isinstance(mapped[0], ABCSeries):
1292     # GH#43986 Need to do list(mapped) in order to get treated as nested
1293     #  See also GH#25959 regarding EA support
1294     return obj._constructor_expanddim(list(mapped), index=obj.index)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/base.py:921, in IndexOpsMixin._map_values(self, mapper, na_action, convert)
918 if isinstance(arr, ExtensionArray):
919     return arr.map(mapper, na_action=na_action)
--> 921 return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/core/algorithms.py:1814, in map_array(arr, mapper, na_action, convert)
1812 values = arr.astype(object, copy=False)
1813 if na_action is None:
-> 1814     return lib.map_infer(values, mapper, convert=convert)
1815 else:
1816     return lib.map_infer_mask(
1817         values, mapper, mask=isna(values).view(np.uint8), convert=convert
1818     )

File lib.pyx:2920, in pandas._libs.lib.map_infer()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/tqdm/std.py:915, in tqdm.pandas.<locals>.inner_generator.<locals>.inner.<locals>.wrapper(*args, **kwargs)
909 def wrapper(*args, **kwargs):
910     # update tbar correctly
911     # it seems `pandas apply` calls `func` twice
912     # on the first column/row to decide whether it can
913     # take a fast or slow code path; so stop when t.total==t.n
914     t.update(n=1 if not t.total or t.n < t.total else 0)
--> 915     return func(*args, **kwargs)

Cell In[21], line 4, in <lambda>(x)
1 import tweetnlp
3 model = tweetnlp.load_model('sentiment')
----> 4 tweetnlp_sentiments = Tweet['cleaned_body'].progress_apply(lambda x: pd.Series(model.sentiment(x, return_probability=True)))
5 tweetnlp_sentiments.head()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/tweetnlp/text_classification/model.py:113, in Classifier.predict(self, text, batch_size, return_probability, skip_preprocess)
106 for i in range(len(_index) - 1):
107     encoded_input = self.tokenizer.batch_encode_plus(
108         text[_index[i]: _index[i+1]],
109         max_length=self.max_length,
110         return_tensors='pt',
111         padding=True,
112         truncation=True)
--> 113     output = self.model(**{k: v.to(self.device) for k, v in encoded_input.items()})
114     if self.multi_label:
115         probs += torch.sigmoid(output.logits).cpu().tolist()

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1511, in Module._wrapped_call_impl(self, *args, **kwargs)
1509     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
1510 else:
-> 1511     return self._call_impl(*args, **kwargs)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1520, in Module._call_impl(self, *args, **kwargs)
1515 # If we don't have any hooks, we want to skip the rest of the logic in
1516 # this function, and just call forward.
1517 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
1518         or _global_backward_pre_hooks or _global_backward_hooks
1519         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1520     return forward_call(*args, **kwargs)
1522 try:
1523     result = None

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/models/roberta/modeling_roberta.py:1206, in RobertaForSequenceClassification.forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
1198 r"""
1199 labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
1200     Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., 1201     config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
1202     `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
1203 """
1204 return_dict = return_dict if return_dict is not None else self.config.use_return_dict
-> 1206 outputs = self.roberta(
1207     input_ids,
1208     attention_mask=attention_mask,
1209     token_type_ids=token_type_ids,
1210     position_ids=position_ids,
1211     head_mask=head_mask,
1212     inputs_embeds=inputs_embeds,
1213     output_attentions=output_attentions,
1214     output_hidden_states=output_hidden_states,
1215     return_dict=return_dict,
1216 )
1217 sequence_output = outputs[0]
1218 logits = self.classifier(sequence_output)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1511, in Module._wrapped_call_impl(self, *args, **kwargs)
1509     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
1510 else:
-> 1511     return self._call_impl(*args, **kwargs)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1520, in Module._call_impl(self, *args, **kwargs)
1515 # If we don't have any hooks, we want to skip the rest of the logic in
1516 # this function, and just call forward.
1517 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
1518         or _global_backward_pre_hooks or _global_backward_hooks
1519         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1520     return forward_call(*args, **kwargs)
1522 try:
1523     result = None

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/models/roberta/modeling_roberta.py:848, in RobertaModel.forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)
839 head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
841 embedding_output = self.embeddings(
842     input_ids=input_ids,
843     position_ids=position_ids,
(...)
846     past_key_values_length=past_key_values_length,
847 )
--> 848 encoder_outputs = self.encoder(
849     embedding_output,
850     attention_mask=extended_attention_mask,
851     head_mask=head_mask,
852     encoder_hidden_states=encoder_hidden_states,
853     encoder_attention_mask=encoder_extended_attention_mask,
854     past_key_values=past_key_values,
855     use_cache=use_cache,
856     output_attentions=output_attentions,
857     output_hidden_states=output_hidden_states,
858     return_dict=return_dict,
859 )
860 sequence_output = encoder_outputs[0]
861 pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1511, in Module._wrapped_call_impl(self, *args, **kwargs)
1509     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
1510 else:
-> 1511     return self._call_impl(*args, **kwargs)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1520, in Module._call_impl(self, *args, **kwargs)
1515 # If we don't have any hooks, we want to skip the rest of the logic in
1516 # this function, and just call forward.
1517 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
1518         or _global_backward_pre_hooks or _global_backward_hooks
1519         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1520     return forward_call(*args, **kwargs)
1522 try:
1523     result = None

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/models/roberta/modeling_roberta.py:524, in RobertaEncoder.forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)
515     layer_outputs = torch.utils.checkpoint.checkpoint(
516         create_custom_forward(layer_module),
517         hidden_states,
(...)
521         encoder_attention_mask,
522     )
523 else:
--> 524     layer_outputs = layer_module(
525         hidden_states,
526         attention_mask,
527         layer_head_mask,
528         encoder_hidden_states,
529         encoder_attention_mask,
530         past_key_value,
531         output_attentions,
532     )
534 hidden_states = layer_outputs[0]
535 if use_cache:

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1511, in Module._wrapped_call_impl(self, *args, **kwargs)
1509     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
1510 else:
-> 1511     return self._call_impl(*args, **kwargs)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1520, in Module._call_impl(self, *args, **kwargs)
1515 # If we don't have any hooks, we want to skip the rest of the logic in
1516 # this function, and just call forward.
1517 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
1518         or _global_backward_pre_hooks or _global_backward_hooks
1519         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1520     return forward_call(*args, **kwargs)
1522 try:
1523     result = None

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/models/roberta/modeling_roberta.py:409, in RobertaLayer.forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
397 def forward(
398     self,
399     hidden_states: torch.Tensor,
(...)
406 ) -> Tuple[torch.Tensor]:
407     # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
408     self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
--> 409     self_attention_outputs = self.attention(
410         hidden_states,
411         attention_mask,
412         head_mask,
413         output_attentions=output_attentions,
414         past_key_value=self_attn_past_key_value,
415     )
416     attention_output = self_attention_outputs[0]
418     # if decoder, the last output is tuple of self-attn cache

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1511, in Module._wrapped_call_impl(self, *args, **kwargs)
1509     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
1510 else:
-> 1511     return self._call_impl(*args, **kwargs)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1520, in Module._call_impl(self, *args, **kwargs)
1515 # If we don't have any hooks, we want to skip the rest of the logic in
1516 # this function, and just call forward.
1517 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
1518         or _global_backward_pre_hooks or _global_backward_hooks
1519         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1520     return forward_call(*args, **kwargs)
1522 try:
1523     result = None

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/models/roberta/modeling_roberta.py:345, in RobertaAttention.forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
326 def forward(
327     self,
328     hidden_states: torch.Tensor,
(...)
334     output_attentions: Optional[bool] = False,
335 ) -> Tuple[torch.Tensor]:
336     self_outputs = self.self(
337         hidden_states,
338         attention_mask,
(...)
343         output_attentions,
344     )
--> 345     attention_output = self.output(self_outputs[0], hidden_states)
346     outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
347     return outputs

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1511, in Module._wrapped_call_impl(self, *args, **kwargs)
1509     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
1510 else:
-> 1511     return self._call_impl(*args, **kwargs)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1520, in Module._call_impl(self, *args, **kwargs)
1515 # If we don't have any hooks, we want to skip the rest of the logic in
1516 # this function, and just call forward.
1517 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
1518         or _global_backward_pre_hooks or _global_backward_hooks
1519         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1520     return forward_call(*args, **kwargs)
1522 try:
1523     result = None

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/models/roberta/modeling_roberta.py:296, in RobertaSelfOutput.forward(self, hidden_states, input_tensor)
294 hidden_states = self.dense(hidden_states)
295 hidden_states = self.dropout(hidden_states)
--> 296 hidden_states = self.LayerNorm(hidden_states + input_tensor)
297 return hidden_states

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1511, in Module._wrapped_call_impl(self, *args, **kwargs)
1509     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
1510 else:
-> 1511     return self._call_impl(*args, **kwargs)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/module.py:1520, in Module._call_impl(self, *args, **kwargs)
1515 # If we don't have any hooks, we want to skip the rest of the logic in
1516 # this function, and just call forward.
1517 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
1518         or _global_backward_pre_hooks or _global_backward_hooks
1519         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1520     return forward_call(*args, **kwargs)
1522 try:
1523     result = None

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/modules/normalization.py:201, in LayerNorm.forward(self, input)
200 def forward(self, input: Tensor) -> Tensor:
--> 201     return F.layer_norm(
202         input, self.normalized_shape, self.weight, self.bias, self.eps)

File /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/torch/nn/functional.py:2546, in layer_norm(input, normalized_shape, weight, bias, eps)
2542 if has_torch_function_variadic(input, weight, bias):
2543     return handle_torch_function(
2544         layer_norm, (input, weight, bias), input, normalized_shape, weight=weight, bias=bias, eps=eps
2545     )
-> 2546 return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)

KeyboardInterrupt:

```python
Tweet = pd.read_csv('Tweet_after_cleaned.csv',index_col=0)
sneti_tweetnlp = pd.read_csv('sneti_tweetnlp.csv',index_col=0).reset_index()
sneti_tweetnlp = sneti_tweetnlp.drop('probability',axis=1)
df = pd.concat([Tweet,sneti_tweetnlp],axis=1)
label_encoding = {'positive': 1, 'negative': -1, 'neutral': 0}
df['encoded_label'] = df['label'].map(label_encoding)

# final_score: encoded_label * (total_engagement + 1)
df['final_score'] = df['encoded_label'] * (df['total_engagement'] + 1)

sneti_tweetnlp_mean = df.groupby(['date', 'ticker_symbol'])['final_score'].mean().reset_index()
sneti_tweetnlp_mean.columns = ['date','ticker_symbol','polarity']
sneti_tweetnlp_mean
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
      <th>date</th>
      <th>ticker_symbol</th>
      <th>polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>-0.164</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01</td>
      <td>AMZN</td>
      <td>-3.557</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01</td>
      <td>GOOG</td>
      <td>0.250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01</td>
      <td>GOOGL</td>
      <td>0.044</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01</td>
      <td>MSFT</td>
      <td>-0.148</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10943</th>
      <td>2019-12-31</td>
      <td>AMZN</td>
      <td>0.120</td>
    </tr>
    <tr>
      <th>10944</th>
      <td>2019-12-31</td>
      <td>GOOG</td>
      <td>-0.246</td>
    </tr>
    <tr>
      <th>10945</th>
      <td>2019-12-31</td>
      <td>GOOGL</td>
      <td>-0.167</td>
    </tr>
    <tr>
      <th>10946</th>
      <td>2019-12-31</td>
      <td>MSFT</td>
      <td>2.087</td>
    </tr>
    <tr>
      <th>10947</th>
      <td>2019-12-31</td>
      <td>TSLA</td>
      <td>-0.195</td>
    </tr>
  </tbody>
</table>
<p>10948 rows × 3 columns</p>
</div>

### 6.2.6. all sentiment results

```python
display('senti_textblob_mean.head()','senti_vader_mean.head()','sneti_tweetnlp_mean.head()')
```

<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>senti_textblob_mean.head()</p><div>
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
      <th>date</th>
      <th>ticker_symbol</th>
      <th>polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>AAPL</td>
      <td>0.735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01</td>
      <td>AMZN</td>
      <td>-0.754</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01</td>
      <td>GOOG</td>
      <td>0.310</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01</td>
      <td>GOOGL</td>
      <td>0.254</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01</td>
      <td>MSFT</td>
      <td>0.071</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>senti_vader_mean.head()</p><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


  
```python
# Merge sentiment aggregates into stock data
stock_data_senti = pd.merge(stock_data, sentiment_df, on=['date', 'ticker_symbol'],how='inner')
# stock_data_senti = stock_data_senti[['date','open','adj close','ma_7','ma_30','volatility_7','volume','ticker_symbol','textblob_polarity','vader_polarity','tweetnlp_polarity']]
# stock_data_senti.columns = ['date','Open','High','Low','Close','Adj Close','Volume','Ticker','P_mean','P_sum','twt_count']
stock_data_senti['date'] = pd.to_datetime(stock_data_senti['date'])
stock_data_senti
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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
      <th>ticker_symbol</th>
      <th>ma_7</th>
      <th>ma_30</th>
      <th>volatility_7</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>target</th>
      <th>textblob_polarity</th>
      <th>vader_polarity</th>
      <th>tweetnlp_polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>289.930</td>
      <td>293.680</td>
      <td>289.520</td>
      <td>293.650</td>
      <td>292.955</td>
      <td>25201400</td>
      <td>AAPL</td>
      <td>300.349</td>
      <td>313.548</td>
      <td>5.029</td>
      <td>2288.000</td>
      <td>466489.000</td>
      <td>0</td>
      <td>0.354</td>
      <td>0.754</td>
      <td>0.583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-12-30</td>
      <td>289.460</td>
      <td>292.690</td>
      <td>285.220</td>
      <td>291.520</td>
      <td>290.830</td>
      <td>36028600</td>
      <td>AAPL</td>
      <td>297.761</td>
      <td>312.359</td>
      <td>4.014</td>
      <td>3256.000</td>
      <td>535824.000</td>
      <td>0</td>
      <td>1.402</td>
      <td>1.694</td>
      <td>1.230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-12-27</td>
      <td>291.120</td>
      <td>293.970</td>
      <td>288.120</td>
      <td>289.800</td>
      <td>289.114</td>
      <td>36566500</td>
      <td>AAPL</td>
      <td>295.849</td>
      <td>311.365</td>
      <td>4.183</td>
      <td>1194.000</td>
      <td>145161.000</td>
      <td>1</td>
      <td>0.316</td>
      <td>0.604</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-26</td>
      <td>284.820</td>
      <td>289.980</td>
      <td>284.700</td>
      <td>289.910</td>
      <td>289.224</td>
      <td>23280300</td>
      <td>AAPL</td>
      <td>294.637</td>
      <td>310.311</td>
      <td>4.537</td>
      <td>2613.000</td>
      <td>477481.000</td>
      <td>0</td>
      <td>0.417</td>
      <td>0.581</td>
      <td>-0.023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-24</td>
      <td>284.690</td>
      <td>284.890</td>
      <td>282.920</td>
      <td>284.270</td>
      <td>283.597</td>
      <td>12119700</td>
      <td>AAPL</td>
      <td>292.419</td>
      <td>309.119</td>
      <td>5.321</td>
      <td>1348.000</td>
      <td>123904.000</td>
      <td>0</td>
      <td>0.812</td>
      <td>0.847</td>
      <td>-0.043</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7539</th>
      <td>2015-01-08</td>
      <td>212.810</td>
      <td>213.800</td>
      <td>210.010</td>
      <td>210.620</td>
      <td>210.620</td>
      <td>3442500</td>
      <td>TSLA</td>
      <td>200.196</td>
      <td>206.351</td>
      <td>7.609</td>
      <td>281.000</td>
      <td>56644.000</td>
      <td>1</td>
      <td>0.132</td>
      <td>0.320</td>
      <td>-0.084</td>
    </tr>
    <tr>
      <th>7540</th>
      <td>2015-01-07</td>
      <td>213.350</td>
      <td>214.780</td>
      <td>209.780</td>
      <td>210.950</td>
      <td>210.950</td>
      <td>2968400</td>
      <td>TSLA</td>
      <td>202.750</td>
      <td>206.146</td>
      <td>7.816</td>
      <td>283.000</td>
      <td>63504.000</td>
      <td>1</td>
      <td>0.167</td>
      <td>0.220</td>
      <td>-0.048</td>
    </tr>
    <tr>
      <th>7541</th>
      <td>2015-01-06</td>
      <td>210.060</td>
      <td>214.200</td>
      <td>204.210</td>
      <td>211.280</td>
      <td>211.280</td>
      <td>6261900</td>
      <td>TSLA</td>
      <td>205.523</td>
      <td>206.131</td>
      <td>6.673</td>
      <td>216.000</td>
      <td>57600.000</td>
      <td>0</td>
      <td>0.140</td>
      <td>0.177</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>7542</th>
      <td>2015-01-05</td>
      <td>214.550</td>
      <td>216.500</td>
      <td>207.160</td>
      <td>210.090</td>
      <td>210.090</td>
      <td>5368500</td>
      <td>TSLA</td>
      <td>208.009</td>
      <td>206.319</td>
      <td>3.653</td>
      <td>309.000</td>
      <td>77284.000</td>
      <td>1</td>
      <td>0.205</td>
      <td>0.271</td>
      <td>-0.007</td>
    </tr>
    <tr>
      <th>7543</th>
      <td>2015-01-02</td>
      <td>222.870</td>
      <td>223.250</td>
      <td>213.260</td>
      <td>219.310</td>
      <td>219.310</td>
      <td>4764400</td>
      <td>TSLA</td>
      <td>210.160</td>
      <td>206.818</td>
      <td>5.184</td>
      <td>218.000</td>
      <td>40401.000</td>
      <td>0</td>
      <td>0.198</td>
      <td>0.242</td>
      <td>0.119</td>
    </tr>
  </tbody>
</table>
<p>7544 rows × 17 columns</p>
</div>

```python
stock_data_senti.isna().sum().sum()
```

0

```python
Tweet.to_csv('Tweet.csv')
stock_data.to_csv('stock_data.csv')
stock_data_senti.to_csv('stock_data_senti.csv')
```

# 5. Model Training and Evaluation

### 5.1. Define Target Variable:

```python
# Define the target variable: stock price movement (1 for up, 0 for unchanged or down)
stock_data_senti['target'] = np.where(stock_data_senti['adj close'].shift(-1) > stock_data_senti['adj close'], 1, 0)

# Drop rows with NaN values created by the shift operation
stock_data_senti = stock_data_senti.dropna()

# Display the first few rows to verify
print("Stock Data with Target Variable:")
stock_data_senti.head()
```

Stock Data with Target Variable:

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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
      <th>ticker_symbol</th>
      <th>ma_7</th>
      <th>ma_30</th>
      <th>volatility_7</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>target</th>
      <th>textblob_polarity</th>
      <th>vader_polarity</th>
      <th>tweetnlp_polarity</th>
      <th>label_t1</th>
      <th>label_t7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>289.930</td>
      <td>293.680</td>
      <td>289.520</td>
      <td>293.650</td>
      <td>292.955</td>
      <td>25201400</td>
      <td>AAPL</td>
      <td>300.349</td>
      <td>313.548</td>
      <td>5.029</td>
      <td>2288.000</td>
      <td>466489.000</td>
      <td>0</td>
      <td>0.354</td>
      <td>0.754</td>
      <td>0.583</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-12-30</td>
      <td>289.460</td>
      <td>292.690</td>
      <td>285.220</td>
      <td>291.520</td>
      <td>290.830</td>
      <td>36028600</td>
      <td>AAPL</td>
      <td>297.761</td>
      <td>312.359</td>
      <td>4.014</td>
      <td>3256.000</td>
      <td>535824.000</td>
      <td>0</td>
      <td>1.402</td>
      <td>1.694</td>
      <td>1.230</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-12-27</td>
      <td>291.120</td>
      <td>293.970</td>
      <td>288.120</td>
      <td>289.800</td>
      <td>289.114</td>
      <td>36566500</td>
      <td>AAPL</td>
      <td>295.849</td>
      <td>311.365</td>
      <td>4.183</td>
      <td>1194.000</td>
      <td>145161.000</td>
      <td>1</td>
      <td>0.316</td>
      <td>0.604</td>
      <td>0.008</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-26</td>
      <td>284.820</td>
      <td>289.980</td>
      <td>284.700</td>
      <td>289.910</td>
      <td>289.224</td>
      <td>23280300</td>
      <td>AAPL</td>
      <td>294.637</td>
      <td>310.311</td>
      <td>4.537</td>
      <td>2613.000</td>
      <td>477481.000</td>
      <td>0</td>
      <td>0.417</td>
      <td>0.581</td>
      <td>-0.023</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-24</td>
      <td>284.690</td>
      <td>284.890</td>
      <td>282.920</td>
      <td>284.270</td>
      <td>283.597</td>
      <td>12119700</td>
      <td>AAPL</td>
      <td>292.419</td>
      <td>309.119</td>
      <td>5.321</td>
      <td>1348.000</td>
      <td>123904.000</td>
      <td>0</td>
      <td>0.812</td>
      <td>0.847</td>
      <td>-0.043</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

### 5.2. Train-Test Split:

```python
# Combine the features from stock_data_senti and Tweet data (assuming we have aligned them by date and ticker_symbol)
# Here, I'll use stock_data_senti as the primary dataset for splitting

# Select feature columns
feature_columns = ['open','adj close','volume','ma_7', 'ma_30', 'volatility_7']

# Add tweet features to feature_columns if they are engineered and aligned
# feature_columns += ['textblob_polarity','vader_polarity','tweetnlp_polarity']

# Split the data into features (X) and target (y)
X = stock_data_senti[feature_columns]
y = stock_data_senti['target']

# Perform a train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Display the shape of the split data
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
```

X_train shape: (5280, 6), X_test shape: (2264, 6)
y_train shape: (5280,), y_test shape: (2264,)

### 5.3. Model Training and Evaluation:

```python
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize XGBoost model
xgboost_model = xgb.XGBClassifier()

# Train XGBoost model
xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_test)

# Evaluate the XGBoost model
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)

# Display the evaluation results for XGBoost
print(f"XGBoost - Precision: {xgb_precision}, Recall: {xgb_recall}, F1 Score: {xgb_f1}")

# (Optional) Evaluate the other models for comparison
# Initialize and train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate Logistic Regression model
logistic_precision = precision_score(y_test, y_pred_logistic)
logistic_recall = recall_score(y_test, y_pred_logistic)
logistic_f1 = f1_score(y_test, y_pred_logistic)

# Initialize and train Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate Random Forest model
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

# Data without Twitter sentiment scores
performance_without_twitter = {
    'Model': ['XGBoost', 'Logistic Regression', 'Random Forest'],
    'Precision': [xgb_precision, logistic_precision, rf_precision],
    'Recall': [xgb_recall, logistic_recall, rf_recall],
    'F1 Score': [xgb_f1, logistic_f1, rf_f1]
}

# Display the evaluation results
print(f"Logistic Regression - Precision: {logistic_precision}, Recall: {logistic_recall}, F1 Score: {logistic_f1}")
print(f"Random Forest - Precision: {rf_precision}, Recall: {rf_recall}, F1 Score: {rf_f1}")
```

XGBoost - Precision: 0.49273743016759775, Recall: 0.40682656826568264, F1 Score: 0.4456796361798888

/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
n_iter_i = _check_optimize_result(

Logistic Regression - Precision: 0.6980552712384852, Recall: 0.6291512915129152, F1 Score: 0.6618146530810286
Random Forest - Precision: 0.48320158102766797, Recall: 0.4511070110701107, F1 Score: 0.4666030534351145

### 5.4. Code for Combining Features

### 5.6. Code for Splitting the Data

```python
# Select feature columns
feature_columns += ['textblob_polarity','vader_polarity','tweetnlp_polarity']

# Split the data into features (X) and target (y)
X = stock_data_senti[feature_columns]
y = stock_data_senti['target']

# Perform a train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# Display the shape of the split data
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
```

X_train shape: (5280, 9), X_test shape: (2264, 9)
y_train shape: (5280,), y_test shape: (2264,)

### 5.7. Model Training and Evaluation

```python
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize XGBoost model
xgboost_model = xgb.XGBClassifier()

# Train XGBoost model
xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_test)

# Evaluate the XGBoost model
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)

# Display the evaluation results for XGBoost
print(f"XGBoost - Precision: {xgb_precision}, Recall: {xgb_recall}, F1 Score: {xgb_f1}")

# (Optional) Evaluate the other models for comparison
# Initialize and train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate Logistic Regression model
logistic_precision = precision_score(y_test, y_pred_logistic)
logistic_recall = recall_score(y_test, y_pred_logistic)
logistic_f1 = f1_score(y_test, y_pred_logistic)

# Initialize and train Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate Random Forest model
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

# Data without Twitter sentiment scores
performance_with_twitter = {
    'Model': ['XGBoost', 'Logistic Regression', 'Random Forest'],
    'Precision': [xgb_precision, logistic_precision, rf_precision],
    'Recall': [xgb_recall, logistic_recall, rf_recall],
    'F1 Score': [xgb_f1, logistic_f1, rf_f1]
}

# Display the evaluation results
print(f"Logistic Regression - Precision: {logistic_precision}, Recall: {logistic_recall}, F1 Score: {logistic_f1}")
print(f"Random Forest - Precision: {rf_precision}, Recall: {rf_recall}, F1 Score: {rf_f1}")
```

XGBoost - Precision: 0.5070821529745042, Recall: 0.49538745387453875, F1 Score: 0.5011665888940737

/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
n_iter_i = _check_optimize_result(

Logistic Regression - Precision: 0.6986721144024515, Recall: 0.6309963099630996, F1 Score: 0.6631119728550654
Random Forest - Precision: 0.496551724137931, Recall: 0.5313653136531366, F1 Score: 0.5133689839572193

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Convert dictionaries to DataFrames
df_without_twitter = pd.DataFrame(performance_without_twitter)
df_with_twitter = pd.DataFrame(performance_with_twitter)

# Add a column to indicate whether the data includes Twitter sentiment scores
df_without_twitter['Twitter'] = 'Without Twitter'
df_with_twitter['Twitter'] = 'With Twitter'

# Concatenate the DataFrames
df_combined = pd.concat([df_without_twitter, df_with_twitter])

# Melt the DataFrame for easier plotting with seaborn
df_melted = df_combined.melt(id_vars=['Model', 'Twitter'], var_name='Metric', value_name='Score')

# Set up the matplotlib figure
plt.figure(figsize=(14, 8))

# Plot the data
sns.barplot(x='Model', y='Score', hue='Twitter', data=df_melted, palette=['#FF9999', '#66B2FF'], edgecolor='black')

# Add titles and labels
plt.title('Model Performance Comparison With and Without Twitter Sentiment Scores')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend(title='Data Source')

# Display the plot
plt.show()
```

![png](tweet_stock_files/tweet_stock_79_0.png)

```python
# Initialize RandomForestClassifier
rf_model = RandomForestClassifier()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best estimator
best_rf_model = grid_search.best_estimator_

# Predict on the test set
y_pred_rf_best = best_rf_model.predict(X_test)

# Evaluate the model
best_rf_precision = precision_score(y_test, y_pred_rf_best)
best_rf_recall = recall_score(y_test, y_pred_rf_best)
best_rf_f1 = f1_score(y_test, y_pred_rf_best)

# Display the evaluation results
print(f"Best Random Forest - Precision: {best_rf_precision}, Recall: {best_rf_recall}, F1 Score: {best_rf_f1}")
```

Fitting 3 folds for each of 108 candidates, totalling 324 fits
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.2s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.1s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.2s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.3s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.6s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.9s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.7s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.6s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.8s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.8s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   4.0s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.2s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.7s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.3s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.3s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.2s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.2s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.2s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.1s
[CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.4s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.0s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.0s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.2s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.0s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.1s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.2s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.2s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.2s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.1s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.2s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.2s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.2s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.4s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.1s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.3s
[CV] END max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   2.9s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   3.1s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   2.9s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.1s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   1.9s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.2s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   0.8s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   1.5s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   3.0s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   3.0s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   1.6s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   1.6s
[CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.7s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   0.9s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   2.4s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   2.4s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   1.6s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   1.6s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   1.6s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   0.9s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   2.5s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   1.6s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   2.4s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   1.6s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   0.7s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   2.4s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   0.9s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   2.2s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   2.4s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   1.7s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   1.6s
[CV] END max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   1.6s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   0.7s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   2.4s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   1.6s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   2.4s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   0.8s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   1.5s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   2.3s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   1.5s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.2s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.4s
[CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.2s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   1.9s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.1s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   2.9s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   1.9s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.1s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.2s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.2s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.2s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.1s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   0.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.0s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.1s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   1.9s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   2.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   1.8s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   0.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   0.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   2.8s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   2.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   1.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   1.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   2.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   1.8s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   0.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   0.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   2.8s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   2.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   1.8s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.0s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   1.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   2.8s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.9s
[CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.7s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.1s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.2s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.1s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.1s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.3s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.2s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.1s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.1s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.2s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.3s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.2s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.5s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.5s
[CV] END max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.5s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.4s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.5s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.4s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.5s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.7s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.1s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.1s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.1s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.1s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.2s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.0s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.2s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   1.9s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   1.9s
[CV] END max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.1s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   0.9s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   3.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   2.9s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   3.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   1.9s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   0.9s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   2.8s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   2.9s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   1.9s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   1.9s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.0s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.4s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.4s
[CV] END max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   2.5s
Best Random Forest - Precision: 0.49448123620309054, Recall: 0.4132841328413284, F1 Score: 0.45025125628140705

```python
# Initial seed money
seed_money = 10000

# Simulate trading based on model predictions
def simulate_trading(predictions, stock_data_senti, seed_money):
    total_money = seed_money
    for i in range(len(predictions)):
        if predictions[i] == 1:  # Model predicts stock price will go up
            # Buy at the opening price and sell at the closing price
            opening_price = stock_data_senti.iloc[i]['open']
            closing_price = stock_data_senti.iloc[i]['adj close']
            if opening_price > 0:  # Avoid division by zero
                total_money = total_money * (closing_price / opening_price)
    return total_money

# Apply the simulation using the best random forest model
final_amount = simulate_trading(y_pred_rf_best, stock_data_senti.loc[X_test.index], seed_money)
profit = final_amount - seed_money

print(f"Final amount after trading: ${final_amount:.2f}")
print(f"Total profit: ${profit:.2f}")
```

Final amount after trading: $0.00
Total profit: $-10000.00

```python
stock_data.to_csv('stock_data.csv')
```

# 지금 해야해야될건

1. 댓글에서 감성분석해서 결과값을 숫자?로 추출
2. 그거를 트레이닝 데이터에 넣기
3. 어떤걸 예측할건지 target 정하기 (주식이 오르고 내릴걸 예측, 주식이 얼마나 오르고 내릴건지 구체적인 가격을 예측할건지)
4. 모델은 어떤걸 쓸건지 LSTM

# 5. Sentiment + Stock Price Viualization

```python
stock_data_senti[['date','textblob_polarity','vader_polarity','tweetnlp_polarity']].head()
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
      <th>date</th>
      <th>textblob_polarity</th>
      <th>vader_polarity</th>
      <th>tweetnlp_polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>0.354</td>
      <td>0.754</td>
      <td>0.583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-12-30</td>
      <td>1.402</td>
      <td>1.694</td>
      <td>1.230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-12-27</td>
      <td>0.316</td>
      <td>0.604</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-26</td>
      <td>0.417</td>
      <td>0.581</td>
      <td>-0.023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-24</td>
      <td>0.812</td>
      <td>0.847</td>
      <td>-0.043</td>
    </tr>
  </tbody>
</table>
</div>

```python
stock_data_senti[['textblob_polarity','vader_polarity','tweetnlp_polarity']].describe()
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
      <th>textblob_polarity</th>
      <th>vader_polarity</th>
      <th>tweetnlp_polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7544.000</td>
      <td>7544.000</td>
      <td>7544.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.239</td>
      <td>0.409</td>
      <td>-0.075</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.257</td>
      <td>0.428</td>
      <td>0.687</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.647</td>
      <td>-2.759</td>
      <td>-7.211</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.102</td>
      <td>0.165</td>
      <td>-0.127</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.183</td>
      <td>0.297</td>
      <td>0.021</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.302</td>
      <td>0.544</td>
      <td>0.160</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.581</td>
      <td>4.879</td>
      <td>4.865</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 감성분석 결과점수(compound)의 평균을 값으로 넣었을때 
def stock_senti_plot(t,key = 'tweetnlp_polarity'):
    df = deepcopy(stock_data_senti[stock_data_senti['ticker_symbol'] == t])
    df['date'] = pd.to_datetime(df['date'])
    plt.figure(figsize=(25,10));
    sns.lineplot(x=df["date"],y=df["adj close"],color='black')
    df['sentiment_analysis']=df[key]
    df['sentiment_analysis']=df['sentiment_analysis'].apply(lambda x: 'pos' if x>0 else 'nue' if x==0 else 'neg')
    sns.scatterplot(x=df["date"],y=df['adj close'],hue=df['sentiment_analysis'],palette=['r','b','g'])
    plt.xticks(rotation=45);
    plt.title(f"Stock market of {t} from Jan-2015 to Sep-2019",fontsize=16);
stock_senti_plot('AAPL')
stock_senti_plot('AMZN')
stock_senti_plot('GOOG')
stock_senti_plot('GOOGL')
stock_senti_plot('MSFT')
stock_senti_plot('TSLA')
```

![png](tweet_stock_files/tweet_stock_87_0.png)

![png](tweet_stock_files/tweet_stock_87_1.png)

![png](tweet_stock_files/tweet_stock_87_2.png)

![png](tweet_stock_files/tweet_stock_87_3.png)

![png](tweet_stock_files/tweet_stock_87_4.png)

![png](tweet_stock_files/tweet_stock_87_5.png)

```python
stock_senti_plot('AAPL','textblob_polarity')
stock_senti_plot('AMZN','textblob_polarity')
stock_senti_plot('GOOG','textblob_polarity')
stock_senti_plot('GOOGL','textblob_polarity')
stock_senti_plot('MSFT','textblob_polarity')
stock_senti_plot('TSLA','textblob_polarity')
```

/var/folders/12/vb6sch5j7lx93jxvhtgjvhyh0000gn/T/ipykernel_61982/347198164.py:9: UserWarning: The palette list has more values (3) than needed (2), which may not be intended.
sns.scatterplot(x=df["date"],y=df['adj close'],hue=df['sentiment_analysis'],palette=['r','b','g'])
/var/folders/12/vb6sch5j7lx93jxvhtgjvhyh0000gn/T/ipykernel_61982/347198164.py:9: UserWarning: The palette list has more values (3) than needed (2), which may not be intended.
sns.scatterplot(x=df["date"],y=df['adj close'],hue=df['sentiment_analysis'],palette=['r','b','g'])
/var/folders/12/vb6sch5j7lx93jxvhtgjvhyh0000gn/T/ipykernel_61982/347198164.py:9: UserWarning: The palette list has more values (3) than needed (2), which may not be intended.
sns.scatterplot(x=df["date"],y=df['adj close'],hue=df['sentiment_analysis'],palette=['r','b','g'])
/var/folders/12/vb6sch5j7lx93jxvhtgjvhyh0000gn/T/ipykernel_61982/347198164.py:9: UserWarning: The palette list has more values (3) than needed (2), which may not be intended.
sns.scatterplot(x=df["date"],y=df['adj close'],hue=df['sentiment_analysis'],palette=['r','b','g'])
/var/folders/12/vb6sch5j7lx93jxvhtgjvhyh0000gn/T/ipykernel_61982/347198164.py:9: UserWarning: The palette list has more values (3) than needed (2), which may not be intended.
sns.scatterplot(x=df["date"],y=df['adj close'],hue=df['sentiment_analysis'],palette=['r','b','g'])
/var/folders/12/vb6sch5j7lx93jxvhtgjvhyh0000gn/T/ipykernel_61982/347198164.py:9: UserWarning: The palette list has more values (3) than needed (2), which may not be intended.
sns.scatterplot(x=df["date"],y=df['adj close'],hue=df['sentiment_analysis'],palette=['r','b','g'])
![png](tweet_stock_files/tweet_stock_88_1.png)

![png](tweet_stock_files/tweet_stock_88_2.png)

![png](tweet_stock_files/tweet_stock_88_3.png)

![png](tweet_stock_files/tweet_stock_88_4.png)

![png](tweet_stock_files/tweet_stock_88_5.png)

![png](tweet_stock_files/tweet_stock_88_6.png)

```python
# value_counts = df['sentiment_analysis'].value_counts()

# plt.figure(figsize=(4, 4))
# plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
# plt.title('Sentiment Analysis Distribution')
# plt.show()
```

# 7. Train_Data_Set

```python
stock_data_senti = pd.read_csv('stock_data_senti.csv',index_col=0)
stock_data_senti.head()
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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
      <th>ticker_symbol</th>
      <th>ma_7</th>
      <th>ma_30</th>
      <th>volatility_7</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>target</th>
      <th>textblob_polarity</th>
      <th>vader_polarity</th>
      <th>tweetnlp_polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>289.930</td>
      <td>293.680</td>
      <td>289.520</td>
      <td>293.650</td>
      <td>292.955</td>
      <td>25201400</td>
      <td>AAPL</td>
      <td>300.349</td>
      <td>313.548</td>
      <td>5.029</td>
      <td>2288.000</td>
      <td>466489.000</td>
      <td>0</td>
      <td>0.354</td>
      <td>0.754</td>
      <td>0.583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-12-30</td>
      <td>289.460</td>
      <td>292.690</td>
      <td>285.220</td>
      <td>291.520</td>
      <td>290.830</td>
      <td>36028600</td>
      <td>AAPL</td>
      <td>297.761</td>
      <td>312.359</td>
      <td>4.014</td>
      <td>3256.000</td>
      <td>535824.000</td>
      <td>0</td>
      <td>1.402</td>
      <td>1.694</td>
      <td>1.230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-12-27</td>
      <td>291.120</td>
      <td>293.970</td>
      <td>288.120</td>
      <td>289.800</td>
      <td>289.114</td>
      <td>36566500</td>
      <td>AAPL</td>
      <td>295.849</td>
      <td>311.365</td>
      <td>4.183</td>
      <td>1194.000</td>
      <td>145161.000</td>
      <td>1</td>
      <td>0.316</td>
      <td>0.604</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-26</td>
      <td>284.820</td>
      <td>289.980</td>
      <td>284.700</td>
      <td>289.910</td>
      <td>289.224</td>
      <td>23280300</td>
      <td>AAPL</td>
      <td>294.637</td>
      <td>310.311</td>
      <td>4.537</td>
      <td>2613.000</td>
      <td>477481.000</td>
      <td>0</td>
      <td>0.417</td>
      <td>0.581</td>
      <td>-0.023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-24</td>
      <td>284.690</td>
      <td>284.890</td>
      <td>282.920</td>
      <td>284.270</td>
      <td>283.597</td>
      <td>12119700</td>
      <td>AAPL</td>
      <td>292.419</td>
      <td>309.119</td>
      <td>5.321</td>
      <td>1348.000</td>
      <td>123904.000</td>
      <td>0</td>
      <td>0.812</td>
      <td>0.847</td>
      <td>-0.043</td>
    </tr>
  </tbody>
</table>
</div>

# 8. Deep Neural Network (LSTM)

## 1. Data Scaling ( MinMax Scaler )

`LSTM uses sigmoid and tanh for acitve function, which are seneitive to magnitude`

```python
# load check point
stock_data_senti = pd.read_csv('stock_data_senti.csv',index_col=0)
stock_data_senti.head()
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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
      <th>ticker_symbol</th>
      <th>ma_7</th>
      <th>ma_30</th>
      <th>volatility_7</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>target</th>
      <th>textblob_polarity</th>
      <th>vader_polarity</th>
      <th>tweetnlp_polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>289.930</td>
      <td>293.680</td>
      <td>289.520</td>
      <td>293.650</td>
      <td>292.955</td>
      <td>25201400</td>
      <td>AAPL</td>
      <td>300.349</td>
      <td>313.548</td>
      <td>5.029</td>
      <td>2288.000</td>
      <td>466489.000</td>
      <td>0</td>
      <td>0.354</td>
      <td>0.754</td>
      <td>0.583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-12-30</td>
      <td>289.460</td>
      <td>292.690</td>
      <td>285.220</td>
      <td>291.520</td>
      <td>290.830</td>
      <td>36028600</td>
      <td>AAPL</td>
      <td>297.761</td>
      <td>312.359</td>
      <td>4.014</td>
      <td>3256.000</td>
      <td>535824.000</td>
      <td>0</td>
      <td>1.402</td>
      <td>1.694</td>
      <td>1.230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-12-27</td>
      <td>291.120</td>
      <td>293.970</td>
      <td>288.120</td>
      <td>289.800</td>
      <td>289.114</td>
      <td>36566500</td>
      <td>AAPL</td>
      <td>295.849</td>
      <td>311.365</td>
      <td>4.183</td>
      <td>1194.000</td>
      <td>145161.000</td>
      <td>1</td>
      <td>0.316</td>
      <td>0.604</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-26</td>
      <td>284.820</td>
      <td>289.980</td>
      <td>284.700</td>
      <td>289.910</td>
      <td>289.224</td>
      <td>23280300</td>
      <td>AAPL</td>
      <td>294.637</td>
      <td>310.311</td>
      <td>4.537</td>
      <td>2613.000</td>
      <td>477481.000</td>
      <td>0</td>
      <td>0.417</td>
      <td>0.581</td>
      <td>-0.023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-24</td>
      <td>284.690</td>
      <td>284.890</td>
      <td>282.920</td>
      <td>284.270</td>
      <td>283.597</td>
      <td>12119700</td>
      <td>AAPL</td>
      <td>292.419</td>
      <td>309.119</td>
      <td>5.321</td>
      <td>1348.000</td>
      <td>123904.000</td>
      <td>0</td>
      <td>0.812</td>
      <td>0.847</td>
      <td>-0.043</td>
    </tr>
  </tbody>
</table>
</div>

```python
train_dates = pd.to_datetime(stock_data_senti['date'])
# df_for_training = stock_data_senti[['ticker_symbol','volume', 
#                                      'textblob_polarity', 'vader_polarity', 'tweetnlp_polarity',
#                                      'open','close']]
df_for_training = stock_data_senti[['ticker_symbol','high','low','adj close','volume',
                                    'ma_7','ma_30','volatility_7','total_engagement','tweet_volume',
                                    'textblob_polarity','vader_polarity','tweetnlp_polarity',
                                    'open','close']]

#Variables for training
# cols = ['Ticker','Volume',
#         'textblob_polarity', 'vader_polarity', 'tweetnlp_polarity',
#         'Open','Close',]
cols = ['Ticker','high','low','adj close','Volume',
        'ma_7','ma_30','volatility_7','total_engagement','tweet_volume',
        'textblob_polarity','vader_polarity','tweetnlp_polarity',
        'Open','Close']
polarity_cols = ['textblob_polarity','vader_polarity','tweetnlp_polarity',]

#Date and volume columns are not used in training.
print(cols)

#New dataframe with only training data - 5 columns
df_for_training.columns = cols
df_for_training[cols[1:]] = df_for_training[cols[1:]].astype(float) # except 'Ticker'
df_for_training.index = train_dates
polarity_iloc = [df_for_training.columns.get_loc(c) for c in polarity_cols]
df_for_training.sort_index(ascending=True,inplace=True)
df_for_training
```

['Ticker', 'high', 'low', 'adj close', 'Volume', 'ma_7', 'ma_30', 'volatility_7', 'total_engagement', 'tweet_volume', 'textblob_polarity', 'vader_polarity', 'tweetnlp_polarity', 'Open', 'Close']

/var/folders/12/vb6sch5j7lx93jxvhtgjvhyh0000gn/T/ipykernel_92874/783989767.py:25: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
df_for_training[cols[1:]] = df_for_training[cols[1:]].astype(float) # except 'Ticker'
/var/folders/12/vb6sch5j7lx93jxvhtgjvhyh0000gn/T/ipykernel_92874/783989767.py:28: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
df_for_training.sort_index(ascending=True,inplace=True)

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
      <th>Ticker</th>
      <th>high</th>
      <th>low</th>
      <th>adj close</th>
      <th>Volume</th>
      <th>ma_7</th>
      <th>ma_30</th>
      <th>volatility_7</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>textblob_polarity</th>
      <th>vader_polarity</th>
      <th>tweetnlp_polarity</th>
      <th>Open</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>TSLA</td>
      <td>223.250</td>
      <td>213.260</td>
      <td>219.310</td>
      <td>4764400.000</td>
      <td>210.160</td>
      <td>206.818</td>
      <td>5.184</td>
      <td>218.000</td>
      <td>40401.000</td>
      <td>0.198</td>
      <td>0.242</td>
      <td>0.119</td>
      <td>222.870</td>
      <td>219.310</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>MSFT</td>
      <td>47.420</td>
      <td>46.540</td>
      <td>41.749</td>
      <td>27913900.000</td>
      <td>46.621</td>
      <td>44.472</td>
      <td>0.641</td>
      <td>90.000</td>
      <td>11449.000</td>
      <td>0.005</td>
      <td>0.276</td>
      <td>-0.037</td>
      <td>46.660</td>
      <td>46.760</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>AAPL</td>
      <td>111.440</td>
      <td>107.350</td>
      <td>100.216</td>
      <td>53204600.000</td>
      <td>108.963</td>
      <td>114.290</td>
      <td>2.388</td>
      <td>2373.000</td>
      <td>751689.000</td>
      <td>0.513</td>
      <td>0.369</td>
      <td>-0.105</td>
      <td>111.390</td>
      <td>109.330</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>AMZN</td>
      <td>314.750</td>
      <td>306.960</td>
      <td>308.520</td>
      <td>2783200.000</td>
      <td>299.031</td>
      <td>325.786</td>
      <td>5.461</td>
      <td>480.000</td>
      <td>30276.000</td>
      <td>1.067</td>
      <td>0.643</td>
      <td>0.063</td>
      <td>312.580</td>
      <td>308.520</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>GOOG</td>
      <td>529.815</td>
      <td>522.665</td>
      <td>523.373</td>
      <td>1447500.000</td>
      <td>503.352</td>
      <td>517.915</td>
      <td>11.026</td>
      <td>94.000</td>
      <td>21025.000</td>
      <td>0.088</td>
      <td>0.299</td>
      <td>-0.145</td>
      <td>527.562</td>
      <td>523.373</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>MSFT</td>
      <td>157.770</td>
      <td>156.450</td>
      <td>156.834</td>
      <td>18369400.000</td>
      <td>159.390</td>
      <td>168.715</td>
      <td>1.641</td>
      <td>927.000</td>
      <td>47524.000</td>
      <td>0.692</td>
      <td>2.049</td>
      <td>2.087</td>
      <td>156.770</td>
      <td>157.700</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>GOOGL</td>
      <td>1340.660</td>
      <td>1332.130</td>
      <td>1339.390</td>
      <td>975700.000</td>
      <td>1383.906</td>
      <td>1445.508</td>
      <td>28.189</td>
      <td>286.000</td>
      <td>14400.000</td>
      <td>0.242</td>
      <td>0.382</td>
      <td>-0.167</td>
      <td>1335.790</td>
      <td>1339.390</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>GOOG</td>
      <td>1338.000</td>
      <td>1329.085</td>
      <td>1337.020</td>
      <td>961800.000</td>
      <td>1382.393</td>
      <td>1445.973</td>
      <td>28.573</td>
      <td>145.000</td>
      <td>12996.000</td>
      <td>0.182</td>
      <td>0.295</td>
      <td>-0.246</td>
      <td>1330.110</td>
      <td>1337.020</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>AMZN</td>
      <td>1853.260</td>
      <td>1832.230</td>
      <td>1847.840</td>
      <td>2506500.000</td>
      <td>1889.083</td>
      <td>1936.160</td>
      <td>20.967</td>
      <td>1413.000</td>
      <td>224676.000</td>
      <td>0.358</td>
      <td>0.724</td>
      <td>0.120</td>
      <td>1842.000</td>
      <td>1847.840</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>AAPL</td>
      <td>293.680</td>
      <td>289.520</td>
      <td>292.955</td>
      <td>25201400.000</td>
      <td>300.349</td>
      <td>313.548</td>
      <td>5.029</td>
      <td>2288.000</td>
      <td>466489.000</td>
      <td>0.354</td>
      <td>0.754</td>
      <td>0.583</td>
      <td>289.930</td>
      <td>293.650</td>
    </tr>
  </tbody>
</table>
<p>7544 rows × 15 columns</p>
</div>

```python
ticker = 'AAPL'
df_for_training_AAPL = df_for_training[df_for_training['Ticker'] == ticker]
print(df_for_training_AAPL.iloc[0, -5:-2])
df_for_training_AAPL.head(8)
```

textblob_polarity    0.513
vader_polarity       0.369
tweetnlp_polarity   -0.105
Name: 2015-01-02 00:00:00, dtype: object

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
      <th>Ticker</th>
      <th>high</th>
      <th>low</th>
      <th>adj close</th>
      <th>Volume</th>
      <th>ma_7</th>
      <th>ma_30</th>
      <th>volatility_7</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>textblob_polarity</th>
      <th>vader_polarity</th>
      <th>tweetnlp_polarity</th>
      <th>Open</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>AAPL</td>
      <td>111.440</td>
      <td>107.350</td>
      <td>100.216</td>
      <td>53204600.000</td>
      <td>108.963</td>
      <td>114.290</td>
      <td>2.388</td>
      <td>2373.000</td>
      <td>751689.000</td>
      <td>0.513</td>
      <td>0.369</td>
      <td>-0.105</td>
      <td>111.390</td>
      <td>109.330</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>AAPL</td>
      <td>108.650</td>
      <td>105.410</td>
      <td>97.393</td>
      <td>64285500.000</td>
      <td>109.090</td>
      <td>114.907</td>
      <td>2.434</td>
      <td>1151.000</td>
      <td>1315609.000</td>
      <td>0.206</td>
      <td>0.297</td>
      <td>-0.059</td>
      <td>108.290</td>
      <td>106.250</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>AAPL</td>
      <td>107.430</td>
      <td>104.630</td>
      <td>97.402</td>
      <td>65797100.000</td>
      <td>109.597</td>
      <td>115.656</td>
      <td>2.089</td>
      <td>815.000</td>
      <td>1416100.000</td>
      <td>0.151</td>
      <td>0.212</td>
      <td>-0.020</td>
      <td>106.540</td>
      <td>106.260</td>
    </tr>
    <tr>
      <th>2015-01-07</th>
      <td>AAPL</td>
      <td>108.200</td>
      <td>106.700</td>
      <td>98.768</td>
      <td>40105900.000</td>
      <td>109.677</td>
      <td>116.395</td>
      <td>1.945</td>
      <td>879.000</td>
      <td>1196836.000</td>
      <td>0.154</td>
      <td>0.224</td>
      <td>-0.038</td>
      <td>107.200</td>
      <td>107.750</td>
    </tr>
    <tr>
      <th>2015-01-08</th>
      <td>AAPL</td>
      <td>112.150</td>
      <td>108.700</td>
      <td>102.563</td>
      <td>59364500.000</td>
      <td>109.426</td>
      <td>117.120</td>
      <td>2.315</td>
      <td>1627.000</td>
      <td>2235025.000</td>
      <td>0.266</td>
      <td>0.377</td>
      <td>0.302</td>
      <td>109.230</td>
      <td>111.890</td>
    </tr>
    <tr>
      <th>2015-01-09</th>
      <td>AAPL</td>
      <td>113.250</td>
      <td>110.210</td>
      <td>102.673</td>
      <td>53699500.000</td>
      <td>108.973</td>
      <td>117.824</td>
      <td>2.047</td>
      <td>1067.000</td>
      <td>1258884.000</td>
      <td>0.215</td>
      <td>0.248</td>
      <td>0.168</td>
      <td>112.670</td>
      <td>112.010</td>
    </tr>
    <tr>
      <th>2015-01-12</th>
      <td>AAPL</td>
      <td>112.630</td>
      <td>108.800</td>
      <td>100.143</td>
      <td>49650800.000</td>
      <td>108.621</td>
      <td>118.496</td>
      <td>1.601</td>
      <td>734.000</td>
      <td>887364.000</td>
      <td>0.147</td>
      <td>0.284</td>
      <td>0.081</td>
      <td>112.600</td>
      <td>109.250</td>
    </tr>
    <tr>
      <th>2015-01-13</th>
      <td>AAPL</td>
      <td>112.800</td>
      <td>108.910</td>
      <td>101.032</td>
      <td>67091900.000</td>
      <td>109.071</td>
      <td>119.147</td>
      <td>2.154</td>
      <td>1146.000</td>
      <td>1882384.000</td>
      <td>0.121</td>
      <td>0.312</td>
      <td>0.060</td>
      <td>111.430</td>
      <td>110.220</td>
    </tr>
  </tbody>
</table>
</div>

```python
def minmaxscaler(df_for_training):  
    scaler = MinMaxScaler()
    df_for_training_scaled = scaler.fit_transform(df_for_training[cols[1:]])

    scaler_for_inference = MinMaxScaler()
    scaler_for_inference.fit_transform(df_for_training.loc[:,['Open','Close']])

    return df_for_training_scaled, scaler_for_inference

df_for_training_scaled, scaler_for_inference = minmaxscaler(df_for_training_AAPL)
print(df_for_training_scaled.shape)
df_for_training_scaled
```

(1255, 14)

array([[0.09772617, 0.08937765, 0.07231139, ..., 0.56774131, 0.10635441,
0.09340419],
[0.08393477, 0.0796801 , 0.0587224 , ..., 0.57534356, 0.09094074,
0.0782549 ],
[0.07790411, 0.07578104, 0.05876665, ..., 0.58185435, 0.08223947,
0.0783041 ],
...,
[1.        , 0.99300178, 0.98151305, ..., 0.58652178, 1.        ,
0.98106337],
[0.99367277, 0.97850543, 0.98977223, ..., 0.78985536, 0.9917462 ,
0.98952336],
[0.99856644, 1.        , 1.        , ..., 0.68220194, 0.99408312,

1.    ]])
## Feature & Label Selection

```python
def trainX_Y(df_for_training_scaled, n_future = 1):
    #Empty lists to be populated using formatted training data
    trainX = []
    trainY = []

    # n_future = 1   # Number of days we want to look into the future based on the past days.
    n_past = 7  # Number of past days we want to use to predict the future.

    #Reformat input data into a shape: (n_samples x timesteps x n_features)
    for i in range(n_past, len(df_for_training_scaled) - n_future): # i = 7 ~ len(df)-7+1
  
        # trainX : day 0~6, 
        #          col   all Columns
        trainX.append(df_for_training_scaled[i - n_past : i])  # ex) 0:7 = day 0~6
  
        # trainY : day 7
        #          col   ['Open','Close']
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, # ex) day 7
                                             -2:]) # ['Open', 'Close]

    trainX, trainY = np.array(trainX), np.array(trainY)
    print('TrainX shape = {}.'.format(trainX.shape))
    print('TrainY shape = {}.'.format(trainY.shape))
    return trainX, trainY

trainX, trainY = trainX_Y(df_for_training_scaled)

```
TrainX shape = (1247, 7, 14).
TrainY shape = (1247, 1, 2).

## 2. Train Test Valid Split

```python
without_polarity = [i for i in range(trainX.shape[-1]) if i+1 not in polarity_iloc]
print(trainX.shape, df_for_training_AAPL.drop('Ticker',axis=1).shape)
df_for_training_AAPL.drop('Ticker',axis=1).columns[without_polarity]
```
(1247, 7, 14) (1255, 14)

Index(['high', 'low', 'adj close', 'Volume', 'ma_7', 'ma_30', 'volatility_7',
'total_engagement', 'tweet_volume', 'Open', 'Close'],
dtype='object')

```python
def train_test_valid_split(trainX, trainY):
    without_polarity = [i for i in range(trainX.shape[-1]) if i+1 not in polarity_iloc] # indexes except polarity data
  
    X_train, X_test, y_train, y_test = train_test_split(trainX[:,:,without_polarity], trainY, test_size=0.2, shuffle=False)

    X_train_twit, X_test_twit, y_train_twit, y_test_twit = train_test_split(trainX, trainY, test_size=0.2, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

    X_train_twit, X_val_twit, y_train_twit, y_val_twit = train_test_split(X_train_twit, y_train_twit, test_size=0.1, shuffle=False)
  
    X = (X_train, X_val, X_test)
    X_t = (X_train_twit, X_val_twit, X_test_twit)
    Y = (y_train, y_val, y_test)
    Y_t = (y_train_twit, y_val_twit, y_test_twit)

    return X, X_t, Y, Y_t

X, X_t, Y, Y_t = train_test_valid_split(trainX, trainY)

V = ['train','val','test']
for T in X, X_t, Y, Y_t:
    for t,v in zip(T,V):
        print(v,':',t.shape)
    print()
```
train : (897, 7, 11)
val : (100, 7, 11)
test : (250, 7, 11)

train : (897, 7, 14)
val : (100, 7, 14)
test : (250, 7, 14)

train : (897, 1, 2)
val : (100, 1, 2)
test : (250, 1, 2)

train : (897, 1, 2)
val : (100, 1, 2)
test : (250, 1, 2)

## 3. LSTM Model

```python
seed = 0

def build_model(input_shape):
    tf.random.set_seed(seed)
    cnn_lstm_model = Sequential()

    cnn_lstm_model.add(Conv1D(filters=128, kernel_size=2, strides=1, padding='valid', input_shape=input_shape))
    cnn_lstm_model.add(MaxPooling1D(pool_size=2, strides=2))

    cnn_lstm_model.add(Conv1D(filters=64, kernel_size=2, strides=1, padding='valid'))
    cnn_lstm_model.add(MaxPooling1D(pool_size=1, strides=2))

    cnn_lstm_model.add(Bidirectional(LSTM(1024, return_sequences=True)))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Bidirectional(LSTM(512, return_sequences=True)))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Bidirectional(LSTM(256, return_sequences=True)))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Bidirectional(LSTM(128, return_sequences=True)))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Bidirectional(LSTM(64, return_sequences=True)))
    cnn_lstm_model.add(Dropout(0.2))

    cnn_lstm_model.add(Dense(32, activation='relu'))


    cnn_lstm_model.add(Dense(trainY.shape[2], activation='relu'))


    cnn_lstm_model.compile(optimizer='adam', loss='mse')
    cnn_lstm_model.summary()
    return cnn_lstm_model
```
```python
from tensorflow.keras.callbacks import EarlyStopping


(X_train, X_val, X_test) = X
(X_train_twit, X_val_twit, X_test_twit) = X_t
(y_train, y_val, y_test) = Y
(y_train_twit, y_val_twit, y_test_twit) = Y_t

# fit the model
def fit_models(X, X_t, Y, Y_t):
    (X_train, X_val, X_test) = X
    (X_train_twit, X_val_twit, X_test_twit) = X_t
    (y_train, y_val, y_test) = Y
    (y_train_twit, y_val_twit, y_test_twit) = Y_t
  
    cnn_model=build_model((X_train.shape[1],X_train.shape[2]))
    cnn_model_twit=build_model((X_train_twit.shape[1],X_train_twit.shape[2]))
  
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = cnn_model.fit(X_train, y_train, 
                            epochs=100, batch_size=64, 
                            validation_data=(X_val, y_val), verbose=1,
                            callbacks=[early_stopping]
                            )
    history_twit = cnn_model_twit.fit(X_train_twit, y_train_twit, 
                                      epochs=100, batch_size=64, 
                                      validation_data=(X_val_twit, y_val_twit), verbose=1, 
                                      callbacks=[early_stopping],
                                      )
    return cnn_model, cnn_model_twit, history, history_twit

cnn_model, cnn_model_twit, history, history_twit = fit_models(X, X_t, Y, Y_t)
```
Model: "sequential_103"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_200 (Conv1D)         (None, 6, 128)            2944

max_pooling1d_200 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_201 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_201 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_515 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_514 (Dropout)       (None, 1, 2048)           0

bidirectional_516 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_515 (Dropout)       (None, 1, 1024)           0

bidirectional_517 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_516 (Dropout)       (None, 1, 512)            0

bidirectional_518 (Bidirec  (None, 1, 256)            656384
tional)

dropout_517 (Dropout)       (None, 1, 256)            0

bidirectional_519 (Bidirec  (None, 1, 128)            164352
tional)

dropout_518 (Dropout)       (None, 1, 128)            0

dense_204 (Dense)           (None, 1, 32)             4128

dense_205 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22878754 (87.28 MB)
Trainable params: 22878754 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Model: "sequential_104"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_202 (Conv1D)         (None, 6, 128)            3712

max_pooling1d_202 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_203 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_203 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_520 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_519 (Dropout)       (None, 1, 2048)           0

bidirectional_521 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_520 (Dropout)       (None, 1, 1024)           0

bidirectional_522 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_521 (Dropout)       (None, 1, 512)            0

bidirectional_523 (Bidirec  (None, 1, 256)            656384
tional)

dropout_522 (Dropout)       (None, 1, 256)            0

bidirectional_524 (Bidirec  (None, 1, 128)            164352
tional)

dropout_523 (Dropout)       (None, 1, 128)            0

dense_206 (Dense)           (None, 1, 32)             4128

dense_207 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22879522 (87.28 MB)
Trainable params: 22879522 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Epoch 1/100
15/15 [==============================] - 27s 487ms/step - loss: 0.0390 - val_loss: 0.1047
Epoch 2/100
15/15 [==============================] - 4s 254ms/step - loss: 0.0147 - val_loss: 0.0133
Epoch 3/100
15/15 [==============================] - 4s 241ms/step - loss: 0.0032 - val_loss: 0.0074
Epoch 4/100
15/15 [==============================] - 4s 241ms/step - loss: 0.0015 - val_loss: 0.0209
Epoch 5/100
15/15 [==============================] - 4s 240ms/step - loss: 0.0029 - val_loss: 0.0198
Epoch 6/100
15/15 [==============================] - 4s 275ms/step - loss: 0.0013 - val_loss: 0.0048
Epoch 7/100
15/15 [==============================] - 7s 480ms/step - loss: 8.3412e-04 - val_loss: 0.0061
Epoch 8/100
15/15 [==============================] - 5s 346ms/step - loss: 7.7827e-04 - val_loss: 0.0027
Epoch 9/100
15/15 [==============================] - 4s 257ms/step - loss: 6.0924e-04 - val_loss: 0.0020
Epoch 10/100
15/15 [==============================] - 5s 358ms/step - loss: 5.8764e-04 - val_loss: 0.0036
Epoch 11/100
15/15 [==============================] - 4s 259ms/step - loss: 5.8325e-04 - val_loss: 0.0019
Epoch 12/100
15/15 [==============================] - 3s 225ms/step - loss: 6.2770e-04 - val_loss: 0.0015
Epoch 13/100
15/15 [==============================] - 3s 218ms/step - loss: 6.0936e-04 - val_loss: 0.0077
Epoch 14/100
15/15 [==============================] - 3s 226ms/step - loss: 0.0011 - val_loss: 0.0021
Epoch 15/100
15/15 [==============================] - 3s 223ms/step - loss: 5.9615e-04 - val_loss: 0.0046
Epoch 16/100
15/15 [==============================] - 4s 241ms/step - loss: 5.0072e-04 - val_loss: 0.0099
Epoch 17/100
15/15 [==============================] - 3s 212ms/step - loss: 0.0013 - val_loss: 0.0031
Epoch 18/100
15/15 [==============================] - 3s 215ms/step - loss: 7.3043e-04 - val_loss: 0.0032
Epoch 19/100
15/15 [==============================] - 3s 202ms/step - loss: 4.9525e-04 - val_loss: 0.0014
Epoch 20/100
15/15 [==============================] - 4s 247ms/step - loss: 6.0071e-04 - val_loss: 0.0020
Epoch 21/100
15/15 [==============================] - 3s 210ms/step - loss: 4.8645e-04 - val_loss: 0.0027
Epoch 22/100
15/15 [==============================] - 3s 220ms/step - loss: 4.5226e-04 - val_loss: 0.0026
Epoch 23/100
15/15 [==============================] - 3s 222ms/step - loss: 4.5462e-04 - val_loss: 0.0031
Epoch 24/100
15/15 [==============================] - 3s 202ms/step - loss: 4.6871e-04 - val_loss: 0.0049
Epoch 25/100
15/15 [==============================] - 4s 270ms/step - loss: 5.4608e-04 - val_loss: 0.0020
Epoch 26/100
15/15 [==============================] - 3s 213ms/step - loss: 3.9564e-04 - val_loss: 0.0042
Epoch 27/100
15/15 [==============================] - 4s 259ms/step - loss: 4.2473e-04 - val_loss: 0.0043
Epoch 28/100
15/15 [==============================] - 7s 460ms/step - loss: 4.0324e-04 - val_loss: 0.0029
Epoch 29/100
15/15 [==============================] - 6s 393ms/step - loss: 3.7798e-04 - val_loss: 0.0010
Epoch 30/100
15/15 [==============================] - 4s 269ms/step - loss: 4.1108e-04 - val_loss: 0.0019
Epoch 31/100
15/15 [==============================] - 4s 241ms/step - loss: 3.7482e-04 - val_loss: 6.3532e-04
Epoch 32/100
15/15 [==============================] - 4s 295ms/step - loss: 4.5021e-04 - val_loss: 0.0042
Epoch 33/100
15/15 [==============================] - 5s 348ms/step - loss: 5.4139e-04 - val_loss: 0.0031
Epoch 34/100
15/15 [==============================] - 5s 314ms/step - loss: 3.5599e-04 - val_loss: 0.0022
Epoch 35/100
15/15 [==============================] - 4s 282ms/step - loss: 3.4033e-04 - val_loss: 0.0020
Epoch 36/100
15/15 [==============================] - 5s 351ms/step - loss: 3.3994e-04 - val_loss: 0.0023
Epoch 37/100
15/15 [==============================] - 5s 294ms/step - loss: 3.6426e-04 - val_loss: 0.0010
Epoch 38/100
15/15 [==============================] - 6s 420ms/step - loss: 3.6060e-04 - val_loss: 8.7010e-04
Epoch 39/100
15/15 [==============================] - 4s 253ms/step - loss: 3.8869e-04 - val_loss: 6.2767e-04
Epoch 40/100
15/15 [==============================] - 4s 258ms/step - loss: 4.9000e-04 - val_loss: 0.0011
Epoch 41/100
15/15 [==============================] - 4s 288ms/step - loss: 3.6505e-04 - val_loss: 7.6717e-04
Epoch 42/100
15/15 [==============================] - 3s 210ms/step - loss: 4.9248e-04 - val_loss: 0.0054
Epoch 43/100
15/15 [==============================] - 3s 167ms/step - loss: 8.2365e-04 - val_loss: 0.0024
Epoch 44/100
15/15 [==============================] - 3s 178ms/step - loss: 4.2898e-04 - val_loss: 0.0022
Epoch 45/100
15/15 [==============================] - 3s 204ms/step - loss: 3.2329e-04 - val_loss: 0.0011
Epoch 46/100
15/15 [==============================] - 3s 218ms/step - loss: 4.9694e-04 - val_loss: 0.0014
Epoch 47/100
15/15 [==============================] - 4s 297ms/step - loss: 3.9870e-04 - val_loss: 5.6828e-04
Epoch 48/100
15/15 [==============================] - 4s 263ms/step - loss: 4.2367e-04 - val_loss: 7.5870e-04
Epoch 49/100
15/15 [==============================] - 3s 175ms/step - loss: 3.9429e-04 - val_loss: 8.3256e-04
Epoch 50/100
15/15 [==============================] - 4s 243ms/step - loss: 3.2119e-04 - val_loss: 0.0011
Epoch 51/100
15/15 [==============================] - 3s 199ms/step - loss: 2.5187e-04 - val_loss: 0.0021
Epoch 52/100
15/15 [==============================] - 3s 214ms/step - loss: 3.1118e-04 - val_loss: 0.0027
Epoch 53/100
15/15 [==============================] - 3s 224ms/step - loss: 3.0562e-04 - val_loss: 0.0017
Epoch 54/100
15/15 [==============================] - 3s 200ms/step - loss: 2.9762e-04 - val_loss: 0.0040
Epoch 55/100
15/15 [==============================] - 3s 168ms/step - loss: 3.4139e-04 - val_loss: 0.0044
Epoch 56/100
15/15 [==============================] - 3s 168ms/step - loss: 2.9152e-04 - val_loss: 0.0033
Epoch 57/100
15/15 [==============================] - 4s 277ms/step - loss: 3.1801e-04 - val_loss: 0.0035
Epoch 1/100
15/15 [==============================] - 77s 4s/step - loss: 0.0435 - val_loss: 0.1450
Epoch 2/100
15/15 [==============================] - 2s 130ms/step - loss: 0.0146 - val_loss: 0.0475
Epoch 3/100
15/15 [==============================] - 3s 222ms/step - loss: 0.0030 - val_loss: 0.0044
Epoch 4/100
15/15 [==============================] - 3s 235ms/step - loss: 0.0022 - val_loss: 0.0168
Epoch 5/100
15/15 [==============================] - 3s 193ms/step - loss: 0.0020 - val_loss: 0.0198
Epoch 6/100
15/15 [==============================] - 3s 166ms/step - loss: 0.0013 - val_loss: 0.0067
Epoch 7/100
15/15 [==============================] - 3s 167ms/step - loss: 7.4949e-04 - val_loss: 0.0057
Epoch 8/100
15/15 [==============================] - 3s 178ms/step - loss: 7.1176e-04 - val_loss: 0.0056
Epoch 9/100
15/15 [==============================] - 3s 177ms/step - loss: 8.5111e-04 - val_loss: 0.0048
Epoch 10/100
15/15 [==============================] - 3s 183ms/step - loss: 6.2988e-04 - val_loss: 0.0061
Epoch 11/100
15/15 [==============================] - 3s 193ms/step - loss: 7.1398e-04 - val_loss: 0.0049
Epoch 12/100
15/15 [==============================] - 3s 206ms/step - loss: 8.1441e-04 - val_loss: 0.0026
Epoch 13/100
15/15 [==============================] - 3s 161ms/step - loss: 7.9868e-04 - val_loss: 0.0063
Epoch 14/100
15/15 [==============================] - 3s 183ms/step - loss: 7.4122e-04 - val_loss: 0.0035
Epoch 15/100
15/15 [==============================] - 3s 195ms/step - loss: 6.5827e-04 - val_loss: 0.0043
Epoch 16/100
15/15 [==============================] - 3s 172ms/step - loss: 5.7191e-04 - val_loss: 0.0146
Epoch 17/100
15/15 [==============================] - 2s 162ms/step - loss: 0.0027 - val_loss: 0.0061
Epoch 18/100
15/15 [==============================] - 3s 184ms/step - loss: 8.0854e-04 - val_loss: 0.0089
Epoch 19/100
15/15 [==============================] - 2s 161ms/step - loss: 6.6639e-04 - val_loss: 0.0039
Epoch 20/100
15/15 [==============================] - 3s 178ms/step - loss: 5.5348e-04 - val_loss: 0.0024
Epoch 21/100
15/15 [==============================] - 2s 167ms/step - loss: 6.2420e-04 - val_loss: 0.0028
Epoch 22/100
15/15 [==============================] - 3s 169ms/step - loss: 5.3393e-04 - val_loss: 0.0044
Epoch 23/100
15/15 [==============================] - 3s 169ms/step - loss: 5.2421e-04 - val_loss: 0.0054
Epoch 24/100
15/15 [==============================] - 2s 160ms/step - loss: 5.7444e-04 - val_loss: 0.0023
Epoch 25/100
15/15 [==============================] - 3s 168ms/step - loss: 6.0011e-04 - val_loss: 0.0023
Epoch 26/100
15/15 [==============================] - 2s 139ms/step - loss: 9.6587e-04 - val_loss: 0.0108
Epoch 27/100
15/15 [==============================] - 2s 153ms/step - loss: 8.0672e-04 - val_loss: 0.0090
Epoch 28/100
15/15 [==============================] - 2s 149ms/step - loss: 0.0013 - val_loss: 0.0033
Epoch 29/100
15/15 [==============================] - 2s 143ms/step - loss: 0.0011 - val_loss: 0.0112
Epoch 30/100
15/15 [==============================] - 2s 163ms/step - loss: 6.8483e-04 - val_loss: 0.0035
Epoch 31/100
15/15 [==============================] - 3s 172ms/step - loss: 7.6844e-04 - val_loss: 0.0093
Epoch 32/100
15/15 [==============================] - 3s 192ms/step - loss: 5.8854e-04 - val_loss: 0.0032
Epoch 33/100
15/15 [==============================] - 2s 150ms/step - loss: 4.6049e-04 - val_loss: 0.0044
Epoch 34/100
15/15 [==============================] - 2s 144ms/step - loss: 4.6984e-04 - val_loss: 0.0048

## 4. Visualization

### 1. Plotting Training and validation loss

```python
def loss_plot(history, history_twit):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    # 첫 번째 플롯: 트위터 감성 분석 없이
    axes[0].plot(history.history['loss'], label='Training loss')
    axes[0].plot(history.history['val_loss'], label='Validation loss')
    axes[0].set_title('Training loss Vs. Validation loss without Twitter sentiment analysis')
    axes[0].legend()

    # 두 번째 플롯: 트위터 감성 분석 포함
    axes[1].plot(history_twit.history['loss'], label='Training loss')
    axes[1].plot(history_twit.history['val_loss'], label='Validation loss')
    axes[1].set_title('Training loss Vs. Validation loss with Twitter sentiment analysis')
    axes[1].legend()

    plt.tight_layout()
    plt.show();

# 예제 사용
loss_plot(history, history_twit)
```
![png](tweet_stock_files/tweet_stock_109_0.png)

### 2. Plotting Prediction Results

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np

def plot_predictions_with_dates(type, twitter, dates, y_actual_lstm, y_pred_lstm):
    predicted_features = ['Open', 'Close']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('LSTM Predictions with and without Twitter Sentiment Analysis')

    errors = []
    for i, predicted_feature in enumerate(predicted_features):
        if twitter:
            title = f'LSTM {type} prediction of {predicted_feature} feature after adding Twitter sentiment analysis'
        else:
            title = f'LSTM {type} prediction of {predicted_feature} feature without Twitter sentiment analysis'

        axes[i].set_title(title)
        sns.lineplot(x=dates, y=y_actual_lstm[:, i], label='Actual', ax=axes[i])
        sns.lineplot(x=dates, y=y_pred_lstm[:, i], label='Predicted', ax=axes[i])

        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(y_actual_lstm[:, i], y_pred_lstm[:, i]))
        errors.append(rmse)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print(f'Root Mean Squared Error (RMSE) for {predicted_features[0]} = {errors[0]}')
    print(f'Root Mean Squared Error (RMSE) for {predicted_features[1]} = {errors[1]}')
    print('Total Root Mean Squared Error (RMSE)', np.sqrt(mean_squared_error(y_actual_lstm, y_pred_lstm)))
```
#### Train Data

##### Computing  accuracy

```python
train_dates= df_for_training_AAPL.index[:X_train.shape[0]]
val_dates= df_for_training_AAPL.index[X_train.shape[0]:X_train.shape[0] + X_val.shape[0]]
test_dates= df_for_training_AAPL.index[-X_test.shape[0]:]

def computing_accuracy(X_, X_twit_,y_):
  
    #Make prediction
    prediction = cnn_model.predict(X_)
    prediction_twit = cnn_model_twit.predict(X_twit_)

    prediction=prediction.reshape(prediction.shape[0], prediction.shape[2])
    prediction_twit=prediction_twit.reshape(prediction_twit.shape[0], prediction_twit.shape[2])

    y_pred = scaler_for_inference.inverse_transform(prediction)
    y_pred_twit = scaler_for_inference.inverse_transform(prediction_twit)

    y_reshaped=y_.reshape(y_.shape[0], y_.shape[2])
    y_actual = scaler_for_inference.inverse_transform(y_reshaped)
  
    return  y_pred, y_pred_twit, y_actual

y_pred, y_pred_twit, y_actual = computing_accuracy(X_train, X_train_twit,y_train)
```
29/29 [==============================] - 2s 10ms/step
29/29 [==============================] - 2s 10ms/step

##### Accuracy without twitter

```python
plot_predictions_with_dates('Training',False,train_dates,y_actual,y_pred)
```
![png](tweet_stock_files/tweet_stock_116_0.png)

Root Mean Squared Error (RMSE) for Open = 4.783395799659464
Root Mean Squared Error (RMSE) for Close = 4.336716296771257
Total Root Mean Squared Error (RMSE) 4.565522073918886

##### Accuracy with the impact of twitter sentiment analysis

```python
plot_predictions_with_dates('Training',True,train_dates,y_actual,y_pred_twit)
```
![png](tweet_stock_files/tweet_stock_118_0.png)

Root Mean Squared Error (RMSE) for Open = 3.058509961819492
Root Mean Squared Error (RMSE) for Close = 3.2630001507739252
Total Root Mean Squared Error (RMSE) 3.1624083520712287

#### Valid Data

##### Computing accuracy

##### Accuracy without twitter

```python
y_pred, y_pred_twit, y_actual = computing_accuracy(X_val, X_val_twit, y_val)
plot_predictions_with_dates('Validation',False,val_dates,y_actual,y_pred)
```
4/4 [==============================] - 0s 11ms/step
4/4 [==============================] - 0s 11ms/step
![png](tweet_stock_files/tweet_stock_122_1.png)

Root Mean Squared Error (RMSE) for Open = 8.435624937150946
Root Mean Squared Error (RMSE) for Close = 9.126866658195894
Total Root Mean Squared Error (RMSE) 8.788044807486214

##### Accuracy with the impact of twitter sentiment analysis

```python
plot_predictions_with_dates('Validation',True,val_dates,y_actual,y_pred_twit)
```
![png](tweet_stock_files/tweet_stock_124_0.png)

Root Mean Squared Error (RMSE) for Open = 6.097256299537074
Root Mean Squared Error (RMSE) for Close = 6.583603362566209
Total Root Mean Squared Error (RMSE) 6.345091316042568

#### Test Data

##### Computing accuracy

##### Accuracy without twitter

```python
y_pred, y_pred_twit, y_actual = computing_accuracy(X_test, X_test_twit,y_test)
plot_predictions_with_dates('Testing',False,test_dates,y_actual,y_pred)
```
8/8 [==============================] - 0s 11ms/step
8/8 [==============================] - 0s 12ms/step
![png](tweet_stock_files/tweet_stock_128_1.png)

Root Mean Squared Error (RMSE) for Open = 16.086336986804856
Root Mean Squared Error (RMSE) for Close = 17.270802566385484
Total Root Mean Squared Error (RMSE) 16.689081145169638

##### Accuracy with the impact of twitter sentiment analysis

```python
plot_predictions_with_dates('Testing',True,test_dates,y_actual,y_pred_twit)
```
![png](tweet_stock_files/tweet_stock_130_0.png)

Root Mean Squared Error (RMSE) for Open = 12.288276138461866
Root Mean Squared Error (RMSE) for Close = 12.694057970844177
Total Root Mean Squared Error (RMSE) 12.492814699303034

# 9. LSTM Pipline

## 9.1 AAPL t-7

```python
def tweet_stock_pipline(df_for_training, ticker='AAPL', n_future = 1):
    df_for_training_scaled, scaler_for_inference = minmaxscaler(df_for_training[df_for_training['Ticker'] == ticker])
    trainX, trainY = trainX_Y(df_for_training_scaled, n_future)
    X, X_t, Y, Y_t = train_test_valid_split(trainX, trainY)
    (X_train, X_val, X_test) = X
    (X_train_twit, X_val_twit, X_test_twit) = X_t
    (y_train, y_val, y_test) = Y
    (y_train_twit, y_val_twit, y_test_twit) = Y_t
    cnn_model, cnn_model_twit, history, history_twit = fit_models(X, X_t, Y, Y_t)
    loss_plot(history,history_twit)
  
    train_dates= df_for_training.index[:X_train.shape[0]]
    val_dates= df_for_training.index[X_train.shape[0]:X_train.shape[0] + X_val.shape[0]]
    test_dates= df_for_training.index[-X_test.shape[0]:]
  
    y_pred, y_pred_twit, y_actual = computing_accuracy(X_train, X_train_twit, y_train)
    plot_predictions_with_dates('Training',False,train_dates,y_actual,y_pred)
    plot_predictions_with_dates('Training',True,train_dates,y_actual,y_pred_twit)
  
    y_pred, y_pred_twit, y_actual = computing_accuracy(X_val, X_val_twit, y_val)
    plot_predictions_with_dates('Validation',False,val_dates,y_actual,y_pred)
    plot_predictions_with_dates('Validation',True,val_dates,y_actual,y_pred_twit)
  
    y_pred, y_pred_twit, y_actual = computing_accuracy(X_test, X_test_twit, y_test)
    plot_predictions_with_dates('Testing',False,test_dates,y_actual,y_pred)
    plot_predictions_with_dates('Testing',True,test_dates,y_actual,y_pred_twit)
  
tweet_stock_pipline(df_for_training, 'AAPL', n_future=7)
```
TrainX shape = (1241, 7, 14).
TrainY shape = (1241, 1, 2).
Model: "sequential_88"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_170 (Conv1D)         (None, 6, 128)            2944

max_pooling1d_170 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_171 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_171 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_440 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_440 (Dropout)       (None, 1, 2048)           0

bidirectional_441 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_441 (Dropout)       (None, 1, 1024)           0

bidirectional_442 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_442 (Dropout)       (None, 1, 512)            0

bidirectional_443 (Bidirec  (None, 1, 256)            656384
tional)

dropout_443 (Dropout)       (None, 1, 256)            0

bidirectional_444 (Bidirec  (None, 1, 128)            164352
tional)

dropout_444 (Dropout)       (None, 1, 128)            0

dense_176 (Dense)           (None, 1, 32)             4128

dense_177 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22878754 (87.28 MB)
Trainable params: 22878754 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Model: "sequential_89"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_172 (Conv1D)         (None, 6, 128)            3712

max_pooling1d_172 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_173 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_173 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_445 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_445 (Dropout)       (None, 1, 2048)           0

bidirectional_446 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_446 (Dropout)       (None, 1, 1024)           0

bidirectional_447 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_447 (Dropout)       (None, 1, 512)            0

bidirectional_448 (Bidirec  (None, 1, 256)            656384
tional)

dropout_448 (Dropout)       (None, 1, 256)            0

bidirectional_449 (Bidirec  (None, 1, 128)            164352
tional)

dropout_449 (Dropout)       (None, 1, 128)            0

dense_178 (Dense)           (None, 1, 32)             4128

dense_179 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22879522 (87.28 MB)
Trainable params: 22879522 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Epoch 1/100
14/14 [==============================] - 16s 493ms/step - loss: 0.0382 - val_loss: 0.0802
Epoch 2/100
14/14 [==============================] - 2s 127ms/step - loss: 0.0091 - val_loss: 0.0271
Epoch 3/100
14/14 [==============================] - 3s 193ms/step - loss: 0.0030 - val_loss: 0.0051
Epoch 4/100
14/14 [==============================] - 2s 173ms/step - loss: 0.0020 - val_loss: 0.0042
Epoch 5/100
14/14 [==============================] - 3s 184ms/step - loss: 0.0016 - val_loss: 0.0072
Epoch 6/100
14/14 [==============================] - 2s 138ms/step - loss: 0.0014 - val_loss: 0.0054
Epoch 7/100
14/14 [==============================] - 2s 130ms/step - loss: 0.0012 - val_loss: 0.0040
Epoch 8/100
14/14 [==============================] - 2s 153ms/step - loss: 0.0011 - val_loss: 0.0046
Epoch 9/100
14/14 [==============================] - 2s 113ms/step - loss: 0.0011 - val_loss: 0.0063
Epoch 10/100
14/14 [==============================] - 2s 110ms/step - loss: 8.8877e-04 - val_loss: 0.0047
Epoch 11/100
14/14 [==============================] - 2s 117ms/step - loss: 8.3995e-04 - val_loss: 0.0053
Epoch 12/100
14/14 [==============================] - 2s 121ms/step - loss: 8.1989e-04 - val_loss: 0.0051
Epoch 13/100
14/14 [==============================] - 1s 101ms/step - loss: 7.8795e-04 - val_loss: 0.0028
Epoch 14/100
14/14 [==============================] - 2s 111ms/step - loss: 7.2455e-04 - val_loss: 0.0024
Epoch 15/100
14/14 [==============================] - 1s 93ms/step - loss: 6.6622e-04 - val_loss: 0.0029
Epoch 16/100
14/14 [==============================] - 1s 96ms/step - loss: 5.9490e-04 - val_loss: 0.0081
Epoch 17/100
14/14 [==============================] - 1s 96ms/step - loss: 8.8025e-04 - val_loss: 0.0019
Epoch 18/100
14/14 [==============================] - 1s 100ms/step - loss: 7.7220e-04 - val_loss: 0.0030
Epoch 19/100
14/14 [==============================] - 1s 90ms/step - loss: 6.0567e-04 - val_loss: 0.0019
Epoch 20/100
14/14 [==============================] - 1s 87ms/step - loss: 7.1675e-04 - val_loss: 0.0034
Epoch 21/100
14/14 [==============================] - 1s 87ms/step - loss: 5.6017e-04 - val_loss: 0.0040
Epoch 22/100
14/14 [==============================] - 1s 84ms/step - loss: 5.5849e-04 - val_loss: 0.0023
Epoch 23/100
14/14 [==============================] - 1s 81ms/step - loss: 5.0863e-04 - val_loss: 0.0024
Epoch 24/100
14/14 [==============================] - 1s 81ms/step - loss: 5.0399e-04 - val_loss: 0.0046
Epoch 25/100
14/14 [==============================] - 1s 85ms/step - loss: 5.5828e-04 - val_loss: 0.0038
Epoch 26/100
14/14 [==============================] - 1s 93ms/step - loss: 5.8278e-04 - val_loss: 0.0013
Epoch 27/100
14/14 [==============================] - 1s 81ms/step - loss: 5.2942e-04 - val_loss: 0.0059
Epoch 28/100
14/14 [==============================] - 1s 82ms/step - loss: 5.8679e-04 - val_loss: 0.0018
Epoch 29/100
14/14 [==============================] - 1s 80ms/step - loss: 6.9632e-04 - val_loss: 0.0030
Epoch 30/100
14/14 [==============================] - 1s 81ms/step - loss: 6.8732e-04 - val_loss: 0.0050
Epoch 31/100
14/14 [==============================] - 1s 82ms/step - loss: 6.1741e-04 - val_loss: 0.0021
Epoch 32/100
14/14 [==============================] - 1s 84ms/step - loss: 5.9622e-04 - val_loss: 0.0056
Epoch 33/100
14/14 [==============================] - 1s 84ms/step - loss: 5.6621e-04 - val_loss: 0.0035
Epoch 34/100
14/14 [==============================] - 1s 81ms/step - loss: 4.5274e-04 - val_loss: 0.0029
Epoch 35/100
14/14 [==============================] - 1s 78ms/step - loss: 4.4136e-04 - val_loss: 0.0019
Epoch 36/100
14/14 [==============================] - 1s 80ms/step - loss: 4.2186e-04 - val_loss: 0.0018
Epoch 1/100
14/14 [==============================] - 10s 220ms/step - loss: 0.0378 - val_loss: 0.1262
Epoch 2/100
14/14 [==============================] - 1s 74ms/step - loss: 0.0106 - val_loss: 0.0272
Epoch 3/100
14/14 [==============================] - 1s 73ms/step - loss: 0.0026 - val_loss: 0.0058
Epoch 4/100
14/14 [==============================] - 1s 72ms/step - loss: 0.0018 - val_loss: 0.0078
Epoch 5/100
14/14 [==============================] - 1s 80ms/step - loss: 0.0014 - val_loss: 0.0056
Epoch 6/100
14/14 [==============================] - 1s 76ms/step - loss: 0.0012 - val_loss: 0.0080
Epoch 7/100
14/14 [==============================] - 1s 73ms/step - loss: 0.0012 - val_loss: 0.0058
Epoch 8/100
14/14 [==============================] - 1s 79ms/step - loss: 0.0013 - val_loss: 0.0087
Epoch 9/100
14/14 [==============================] - 1s 82ms/step - loss: 0.0011 - val_loss: 0.0058
Epoch 10/100
14/14 [==============================] - 1s 84ms/step - loss: 0.0011 - val_loss: 0.0054
Epoch 11/100
14/14 [==============================] - 1s 75ms/step - loss: 0.0010 - val_loss: 0.0058
Epoch 12/100
14/14 [==============================] - 1s 76ms/step - loss: 9.8433e-04 - val_loss: 0.0064
Epoch 13/100
14/14 [==============================] - 1s 77ms/step - loss: 8.6934e-04 - val_loss: 0.0046
Epoch 14/100
14/14 [==============================] - 1s 83ms/step - loss: 9.1012e-04 - val_loss: 0.0045
Epoch 15/100
14/14 [==============================] - 1s 75ms/step - loss: 8.4255e-04 - val_loss: 0.0053
Epoch 16/100
14/14 [==============================] - 1s 76ms/step - loss: 8.3677e-04 - val_loss: 0.0077
Epoch 17/100
14/14 [==============================] - 1s 110ms/step - loss: 0.0010 - val_loss: 0.0051
Epoch 18/100
14/14 [==============================] - 1s 76ms/step - loss: 9.2303e-04 - val_loss: 0.0040
Epoch 19/100
14/14 [==============================] - 1s 83ms/step - loss: 7.6480e-04 - val_loss: 0.0036
Epoch 20/100
14/14 [==============================] - 1s 70ms/step - loss: 0.0010 - val_loss: 0.0046
Epoch 21/100
14/14 [==============================] - 1s 72ms/step - loss: 9.5536e-04 - val_loss: 0.0042
Epoch 22/100
14/14 [==============================] - 1s 73ms/step - loss: 7.5522e-04 - val_loss: 0.0033
Epoch 23/100
14/14 [==============================] - 1s 77ms/step - loss: 6.4751e-04 - val_loss: 0.0034
Epoch 24/100
14/14 [==============================] - 1s 72ms/step - loss: 6.7744e-04 - val_loss: 0.0048
Epoch 25/100
14/14 [==============================] - 1s 72ms/step - loss: 6.6513e-04 - val_loss: 0.0040
Epoch 26/100
14/14 [==============================] - 1s 73ms/step - loss: 6.1550e-04 - val_loss: 0.0017
Epoch 27/100
14/14 [==============================] - 1s 71ms/step - loss: 5.9253e-04 - val_loss: 0.0062
Epoch 28/100
14/14 [==============================] - 1s 74ms/step - loss: 6.2297e-04 - val_loss: 0.0019
Epoch 29/100
14/14 [==============================] - 1s 73ms/step - loss: 6.1165e-04 - val_loss: 0.0032
Epoch 30/100
14/14 [==============================] - 1s 72ms/step - loss: 6.4400e-04 - val_loss: 0.0021
Epoch 31/100
14/14 [==============================] - 1s 72ms/step - loss: 6.6116e-04 - val_loss: 0.0028
Epoch 32/100
14/14 [==============================] - 1s 78ms/step - loss: 7.1547e-04 - val_loss: 0.0054
Epoch 33/100
14/14 [==============================] - 1s 73ms/step - loss: 6.3820e-04 - val_loss: 0.0037
Epoch 34/100
14/14 [==============================] - 1s 73ms/step - loss: 5.0226e-04 - val_loss: 0.0024
Epoch 35/100
14/14 [==============================] - 1s 74ms/step - loss: 4.8580e-04 - val_loss: 0.0027
Epoch 36/100
14/14 [==============================] - 1s 87ms/step - loss: 4.7021e-04 - val_loss: 0.0015
Epoch 37/100
14/14 [==============================] - 1s 75ms/step - loss: 5.6340e-04 - val_loss: 0.0016
Epoch 38/100
14/14 [==============================] - 1s 76ms/step - loss: 7.1223e-04 - val_loss: 0.0043
Epoch 39/100
14/14 [==============================] - 1s 77ms/step - loss: 5.7768e-04 - val_loss: 0.0039
Epoch 40/100
14/14 [==============================] - 1s 76ms/step - loss: 4.6236e-04 - val_loss: 0.0020
Epoch 41/100
14/14 [==============================] - 1s 83ms/step - loss: 4.3103e-04 - val_loss: 0.0027
Epoch 42/100
14/14 [==============================] - 1s 77ms/step - loss: 4.2759e-04 - val_loss: 0.0039
Epoch 43/100
14/14 [==============================] - 1s 76ms/step - loss: 5.1715e-04 - val_loss: 0.0021
Epoch 44/100
14/14 [==============================] - 1s 77ms/step - loss: 4.5575e-04 - val_loss: 0.0014
Epoch 45/100
14/14 [==============================] - 1s 83ms/step - loss: 4.4840e-04 - val_loss: 0.0025
Epoch 46/100
14/14 [==============================] - 1s 77ms/step - loss: 4.4972e-04 - val_loss: 0.0011
Epoch 47/100
14/14 [==============================] - 1s 75ms/step - loss: 4.3646e-04 - val_loss: 0.0029
Epoch 48/100
14/14 [==============================] - 1s 77ms/step - loss: 5.2656e-04 - val_loss: 0.0018
Epoch 49/100
14/14 [==============================] - 1s 80ms/step - loss: 6.0833e-04 - val_loss: 0.0027
Epoch 50/100
14/14 [==============================] - 1s 74ms/step - loss: 6.1632e-04 - val_loss: 0.0017
Epoch 51/100
14/14 [==============================] - 1s 74ms/step - loss: 4.7198e-04 - val_loss: 0.0015
Epoch 52/100
14/14 [==============================] - 1s 81ms/step - loss: 4.6784e-04 - val_loss: 0.0024
Epoch 53/100
14/14 [==============================] - 1s 75ms/step - loss: 4.5284e-04 - val_loss: 0.0016
Epoch 54/100
14/14 [==============================] - 1s 77ms/step - loss: 4.6828e-04 - val_loss: 0.0020
Epoch 55/100
14/14 [==============================] - 1s 79ms/step - loss: 4.0487e-04 - val_loss: 0.0013
Epoch 56/100
14/14 [==============================] - 1s 78ms/step - loss: 4.4818e-04 - val_loss: 9.7306e-04
Epoch 57/100
14/14 [==============================] - 1s 75ms/step - loss: 4.6957e-04 - val_loss: 0.0045
Epoch 58/100
14/14 [==============================] - 1s 76ms/step - loss: 4.5679e-04 - val_loss: 0.0019
Epoch 59/100
14/14 [==============================] - 1s 74ms/step - loss: 4.4395e-04 - val_loss: 0.0012
Epoch 60/100
14/14 [==============================] - 1s 106ms/step - loss: 3.9575e-04 - val_loss: 0.0027
Epoch 61/100
14/14 [==============================] - 1s 85ms/step - loss: 4.0835e-04 - val_loss: 0.0027
Epoch 62/100
14/14 [==============================] - 1s 76ms/step - loss: 4.0260e-04 - val_loss: 0.0019
Epoch 63/100
14/14 [==============================] - 1s 76ms/step - loss: 4.2160e-04 - val_loss: 0.0039
Epoch 64/100
14/14 [==============================] - 1s 82ms/step - loss: 3.9584e-04 - val_loss: 9.5216e-04
Epoch 65/100
14/14 [==============================] - 1s 84ms/step - loss: 4.8672e-04 - val_loss: 0.0015
Epoch 66/100
14/14 [==============================] - 1s 77ms/step - loss: 5.0770e-04 - val_loss: 0.0020
Epoch 67/100
14/14 [==============================] - 1s 78ms/step - loss: 4.2260e-04 - val_loss: 0.0024
Epoch 68/100
14/14 [==============================] - 1s 76ms/step - loss: 4.0336e-04 - val_loss: 0.0034
Epoch 69/100
14/14 [==============================] - 1s 84ms/step - loss: 5.3577e-04 - val_loss: 0.0034
Epoch 70/100
14/14 [==============================] - 1s 78ms/step - loss: 4.9439e-04 - val_loss: 0.0016
Epoch 71/100
14/14 [==============================] - 1s 80ms/step - loss: 4.2931e-04 - val_loss: 9.0824e-04
Epoch 72/100
14/14 [==============================] - 1s 78ms/step - loss: 4.2370e-04 - val_loss: 0.0035
Epoch 73/100
14/14 [==============================] - 1s 78ms/step - loss: 5.2962e-04 - val_loss: 0.0014
Epoch 74/100
14/14 [==============================] - 1s 75ms/step - loss: 4.8003e-04 - val_loss: 0.0040
Epoch 75/100
14/14 [==============================] - 1s 75ms/step - loss: 4.4913e-04 - val_loss: 0.0017
Epoch 76/100
14/14 [==============================] - 1s 75ms/step - loss: 4.4565e-04 - val_loss: 0.0014
Epoch 77/100
14/14 [==============================] - 1s 75ms/step - loss: 4.0251e-04 - val_loss: 0.0032
Epoch 78/100
14/14 [==============================] - 1s 82ms/step - loss: 3.9362e-04 - val_loss: 0.0020
Epoch 79/100
14/14 [==============================] - 1s 79ms/step - loss: 3.8290e-04 - val_loss: 9.5867e-04
Epoch 80/100
14/14 [==============================] - 1s 76ms/step - loss: 4.7601e-04 - val_loss: 0.0023
Epoch 81/100
14/14 [==============================] - 1s 79ms/step - loss: 4.0361e-04 - val_loss: 0.0023
![png](tweet_stock_files/tweet_stock_133_1.png)

28/28 [==============================] - 2s 14ms/step
28/28 [==============================] - 1s 11ms/step
![png](tweet_stock_files/tweet_stock_133_3.png)

Root Mean Squared Error (RMSE) for Open = 5.743746178987279
Root Mean Squared Error (RMSE) for Close = 5.52367678191464
Total Root Mean Squared Error (RMSE) 5.634785948006092
![png](tweet_stock_files/tweet_stock_133_5.png)

Root Mean Squared Error (RMSE) for Open = 4.568017222886829
Root Mean Squared Error (RMSE) for Close = 4.804906231374995
Total Root Mean Squared Error (RMSE) 4.687958257114548
4/4 [==============================] - 0s 14ms/step
4/4 [==============================] - 0s 11ms/step
![png](tweet_stock_files/tweet_stock_133_7.png)

Root Mean Squared Error (RMSE) for Open = 12.506857009495327
Root Mean Squared Error (RMSE) for Close = 12.856893966445757
Total Root Mean Squared Error (RMSE) 12.683083117294306
![png](tweet_stock_files/tweet_stock_133_9.png)

Root Mean Squared Error (RMSE) for Open = 10.599413744772276
Root Mean Squared Error (RMSE) for Close = 11.227957454388093
Total Root Mean Squared Error (RMSE) 10.918209567745452
8/8 [==============================] - 0s 12ms/step
8/8 [==============================] - 0s 11ms/step
![png](tweet_stock_files/tweet_stock_133_11.png)

Root Mean Squared Error (RMSE) for Open = 17.248585918354134
Root Mean Squared Error (RMSE) for Close = 18.504720754600594
Total Root Mean Squared Error (RMSE) 17.88768300239847
![png](tweet_stock_files/tweet_stock_133_13.png)

Root Mean Squared Error (RMSE) for Open = 13.89791155931617
Root Mean Squared Error (RMSE) for Close = 14.232765561473645
Total Root Mean Squared Error (RMSE) 14.066335010201561

## 9.2 MSFT t-1

```python
tweet_stock_pipline(df_for_training, 'MSFT')
```
TrainX shape = (1250, 7, 14).
TrainY shape = (1250, 1, 2).
Model: "sequential_90"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_174 (Conv1D)         (None, 6, 128)            2944

max_pooling1d_174 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_175 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_175 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_450 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_450 (Dropout)       (None, 1, 2048)           0

bidirectional_451 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_451 (Dropout)       (None, 1, 1024)           0

bidirectional_452 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_452 (Dropout)       (None, 1, 512)            0

bidirectional_453 (Bidirec  (None, 1, 256)            656384
tional)

dropout_453 (Dropout)       (None, 1, 256)            0

bidirectional_454 (Bidirec  (None, 1, 128)            164352
tional)

dropout_454 (Dropout)       (None, 1, 128)            0

dense_180 (Dense)           (None, 1, 32)             4128

dense_181 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22878754 (87.28 MB)
Trainable params: 22878754 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Model: "sequential_91"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_176 (Conv1D)         (None, 6, 128)            3712

max_pooling1d_176 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_177 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_177 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_455 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_455 (Dropout)       (None, 1, 2048)           0

bidirectional_456 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_456 (Dropout)       (None, 1, 1024)           0

bidirectional_457 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_457 (Dropout)       (None, 1, 512)            0

bidirectional_458 (Bidirec  (None, 1, 256)            656384
tional)

dropout_458 (Dropout)       (None, 1, 256)            0

bidirectional_459 (Bidirec  (None, 1, 128)            164352
tional)

dropout_459 (Dropout)       (None, 1, 128)            0

dense_182 (Dense)           (None, 1, 32)             4128

dense_183 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22879522 (87.28 MB)
Trainable params: 22879522 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Epoch 1/100
15/15 [==============================] - 9s 173ms/step - loss: 0.0376 - val_loss: 0.0881
Epoch 2/100
15/15 [==============================] - 1s 55ms/step - loss: 0.0069 - val_loss: 0.0027
Epoch 3/100
15/15 [==============================] - 1s 56ms/step - loss: 0.0021 - val_loss: 0.0098
Epoch 4/100
15/15 [==============================] - 1s 61ms/step - loss: 0.0012 - val_loss: 0.0012
Epoch 5/100
15/15 [==============================] - 1s 57ms/step - loss: 8.3079e-04 - val_loss: 0.0013
Epoch 6/100
15/15 [==============================] - 1s 58ms/step - loss: 6.0641e-04 - val_loss: 0.0016
Epoch 7/100
15/15 [==============================] - 1s 59ms/step - loss: 5.1322e-04 - val_loss: 0.0011
Epoch 8/100
15/15 [==============================] - 1s 62ms/step - loss: 5.2271e-04 - val_loss: 0.0013
Epoch 9/100
15/15 [==============================] - 1s 61ms/step - loss: 5.1358e-04 - val_loss: 0.0013
Epoch 10/100
15/15 [==============================] - 1s 63ms/step - loss: 4.9530e-04 - val_loss: 7.1649e-04
Epoch 11/100
15/15 [==============================] - 1s 64ms/step - loss: 6.8605e-04 - val_loss: 0.0051
Epoch 12/100
15/15 [==============================] - 1s 68ms/step - loss: 7.8771e-04 - val_loss: 0.0014
Epoch 13/100
15/15 [==============================] - 1s 62ms/step - loss: 6.1871e-04 - val_loss: 8.0326e-04
Epoch 14/100
15/15 [==============================] - 1s 63ms/step - loss: 4.1170e-04 - val_loss: 7.2206e-04
Epoch 15/100
15/15 [==============================] - 1s 64ms/step - loss: 5.1086e-04 - val_loss: 0.0013
Epoch 16/100
15/15 [==============================] - 1s 68ms/step - loss: 5.9071e-04 - val_loss: 6.3764e-04
Epoch 17/100
15/15 [==============================] - 1s 62ms/step - loss: 4.3300e-04 - val_loss: 0.0020
Epoch 18/100
15/15 [==============================] - 1s 64ms/step - loss: 4.6993e-04 - val_loss: 0.0016
Epoch 19/100
15/15 [==============================] - 1s 63ms/step - loss: 4.3697e-04 - val_loss: 0.0012
Epoch 20/100
15/15 [==============================] - 1s 64ms/step - loss: 4.6137e-04 - val_loss: 0.0013
Epoch 21/100
15/15 [==============================] - 1s 70ms/step - loss: 6.1036e-04 - val_loss: 0.0091
Epoch 22/100
15/15 [==============================] - 1s 64ms/step - loss: 0.0018 - val_loss: 0.0079
Epoch 23/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0019 - val_loss: 0.0021
Epoch 24/100
15/15 [==============================] - 1s 67ms/step - loss: 5.2499e-04 - val_loss: 0.0016
Epoch 25/100
15/15 [==============================] - 1s 67ms/step - loss: 3.8602e-04 - val_loss: 0.0010
Epoch 26/100
15/15 [==============================] - 1s 75ms/step - loss: 3.4049e-04 - val_loss: 7.3560e-04
Epoch 1/100
15/15 [==============================] - 10s 211ms/step - loss: 0.0527 - val_loss: 0.2032
Epoch 2/100
15/15 [==============================] - 1s 66ms/step - loss: 0.0360 - val_loss: 0.1752
Epoch 3/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0330 - val_loss: 0.1672
Epoch 4/100
15/15 [==============================] - 1s 80ms/step - loss: 0.0317 - val_loss: 0.1285
Epoch 5/100
15/15 [==============================] - 1s 87ms/step - loss: 0.0056 - val_loss: 0.0196
Epoch 6/100
15/15 [==============================] - 1s 92ms/step - loss: 0.0021 - val_loss: 8.4743e-04
Epoch 7/100
15/15 [==============================] - 1s 77ms/step - loss: 9.3703e-04 - val_loss: 0.0083
Epoch 8/100
15/15 [==============================] - 1s 85ms/step - loss: 8.0207e-04 - val_loss: 9.6430e-04
Epoch 9/100
15/15 [==============================] - 1s 83ms/step - loss: 6.8763e-04 - val_loss: 0.0059
Epoch 10/100
15/15 [==============================] - 1s 81ms/step - loss: 6.2677e-04 - val_loss: 6.2803e-04
Epoch 11/100
15/15 [==============================] - 1s 86ms/step - loss: 5.2121e-04 - val_loss: 8.4220e-04
Epoch 12/100
15/15 [==============================] - 1s 78ms/step - loss: 4.1560e-04 - val_loss: 9.2757e-04
Epoch 13/100
15/15 [==============================] - 1s 76ms/step - loss: 4.7391e-04 - val_loss: 8.3319e-04
Epoch 14/100
15/15 [==============================] - 1s 77ms/step - loss: 4.6138e-04 - val_loss: 5.9139e-04
Epoch 15/100
15/15 [==============================] - 1s 94ms/step - loss: 5.6536e-04 - val_loss: 9.5365e-04
Epoch 16/100
15/15 [==============================] - 1s 83ms/step - loss: 5.0898e-04 - val_loss: 0.0031
Epoch 17/100
15/15 [==============================] - 1s 77ms/step - loss: 5.9572e-04 - val_loss: 0.0024
Epoch 18/100
15/15 [==============================] - 1s 77ms/step - loss: 4.4397e-04 - val_loss: 0.0012
Epoch 19/100
15/15 [==============================] - 1s 87ms/step - loss: 4.2310e-04 - val_loss: 8.4492e-04
Epoch 20/100
15/15 [==============================] - 1s 79ms/step - loss: 3.8849e-04 - val_loss: 8.4149e-04
Epoch 21/100
15/15 [==============================] - 1s 78ms/step - loss: 6.5190e-04 - val_loss: 0.0026
Epoch 22/100
15/15 [==============================] - 1s 100ms/step - loss: 5.0706e-04 - val_loss: 0.0034
Epoch 23/100
15/15 [==============================] - 1s 81ms/step - loss: 6.4847e-04 - val_loss: 6.5591e-04
Epoch 24/100
15/15 [==============================] - 1s 81ms/step - loss: 5.1908e-04 - val_loss: 5.8642e-04
Epoch 25/100
15/15 [==============================] - 1s 89ms/step - loss: 3.9417e-04 - val_loss: 5.4990e-04
Epoch 26/100
15/15 [==============================] - 1s 72ms/step - loss: 3.5085e-04 - val_loss: 9.3410e-04
Epoch 27/100
15/15 [==============================] - 1s 72ms/step - loss: 3.8543e-04 - val_loss: 7.7335e-04
Epoch 28/100
15/15 [==============================] - 1s 72ms/step - loss: 4.0840e-04 - val_loss: 0.0011
Epoch 29/100
15/15 [==============================] - 1s 80ms/step - loss: 4.1151e-04 - val_loss: 9.0254e-04
Epoch 30/100
15/15 [==============================] - 1s 73ms/step - loss: 5.2403e-04 - val_loss: 9.5363e-04
Epoch 31/100
15/15 [==============================] - 1s 73ms/step - loss: 7.3523e-04 - val_loss: 0.0114
Epoch 32/100
15/15 [==============================] - 1s 74ms/step - loss: 6.7375e-04 - val_loss: 6.2629e-04
Epoch 33/100
15/15 [==============================] - 1s 83ms/step - loss: 4.4297e-04 - val_loss: 9.4322e-04
Epoch 34/100
15/15 [==============================] - 1s 74ms/step - loss: 3.9804e-04 - val_loss: 0.0018
Epoch 35/100
15/15 [==============================] - 1s 77ms/step - loss: 3.0100e-04 - val_loss: 0.0013
![png](tweet_stock_files/tweet_stock_135_1.png)

29/29 [==============================] - 1s 10ms/step
29/29 [==============================] - 1s 10ms/step
![png](tweet_stock_files/tweet_stock_135_3.png)

Root Mean Squared Error (RMSE) for Open = 5.748515184896584
Root Mean Squared Error (RMSE) for Close = 4.180458981799437
Total Root Mean Squared Error (RMSE) 5.026015525716876
![png](tweet_stock_files/tweet_stock_135_5.png)

Root Mean Squared Error (RMSE) for Open = 6.940357844402308
Root Mean Squared Error (RMSE) for Close = 6.024849254016295
Total Root Mean Squared Error (RMSE) 6.498745092014971
4/4 [==============================] - 0s 10ms/step
4/4 [==============================] - 0s 10ms/step
![png](tweet_stock_files/tweet_stock_135_7.png)

Root Mean Squared Error (RMSE) for Open = 6.107993578890187
Root Mean Squared Error (RMSE) for Close = 5.382652136814171
Total Root Mean Squared Error (RMSE) 5.756758184330566
![png](tweet_stock_files/tweet_stock_135_9.png)

Root Mean Squared Error (RMSE) for Open = 7.500519150209663
Root Mean Squared Error (RMSE) for Close = 6.601788308137735
Total Root Mean Squared Error (RMSE) 7.065458102208448
8/8 [==============================] - 0s 11ms/step
8/8 [==============================] - 0s 11ms/step
![png](tweet_stock_files/tweet_stock_135_11.png)

Root Mean Squared Error (RMSE) for Open = 23.078828206594554
Root Mean Squared Error (RMSE) for Close = 27.261099706301362
Total Root Mean Squared Error (RMSE) 25.256680983320127
![png](tweet_stock_files/tweet_stock_135_13.png)

Root Mean Squared Error (RMSE) for Open = 16.028172273537777
Root Mean Squared Error (RMSE) for Close = 18.109340935407406
Total Root Mean Squared Error (RMSE) 17.10044642026967

## 9.3 GOOG t-1

```python
tweet_stock_pipline(df_for_training, 'GOOG')
```
TrainX shape = (1249, 7, 14).
TrainY shape = (1249, 1, 2).
Model: "sequential_92"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_178 (Conv1D)         (None, 6, 128)            2944

max_pooling1d_178 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_179 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_179 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_460 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_460 (Dropout)       (None, 1, 2048)           0

bidirectional_461 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_461 (Dropout)       (None, 1, 1024)           0

bidirectional_462 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_462 (Dropout)       (None, 1, 512)            0

bidirectional_463 (Bidirec  (None, 1, 256)            656384
tional)

dropout_463 (Dropout)       (None, 1, 256)            0

bidirectional_464 (Bidirec  (None, 1, 128)            164352
tional)

dropout_464 (Dropout)       (None, 1, 128)            0

dense_184 (Dense)           (None, 1, 32)             4128

dense_185 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22878754 (87.28 MB)
Trainable params: 22878754 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Model: "sequential_93"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_180 (Conv1D)         (None, 6, 128)            3712

max_pooling1d_180 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_181 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_181 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_465 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_465 (Dropout)       (None, 1, 2048)           0

bidirectional_466 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_466 (Dropout)       (None, 1, 1024)           0

bidirectional_467 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_467 (Dropout)       (None, 1, 512)            0

bidirectional_468 (Bidirec  (None, 1, 256)            656384
tional)

dropout_468 (Dropout)       (None, 1, 256)            0

bidirectional_469 (Bidirec  (None, 1, 128)            164352
tional)

dropout_469 (Dropout)       (None, 1, 128)            0

dense_186 (Dense)           (None, 1, 32)             4128

dense_187 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22879522 (87.28 MB)
Trainable params: 22879522 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Epoch 1/100
15/15 [==============================] - 9s 196ms/step - loss: 0.0974 - val_loss: 0.1839
Epoch 2/100
15/15 [==============================] - 1s 66ms/step - loss: 0.0284 - val_loss: 0.0594
Epoch 3/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0104 - val_loss: 0.0187
Epoch 4/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0036 - val_loss: 0.0027
Epoch 5/100
15/15 [==============================] - 1s 82ms/step - loss: 0.0023 - val_loss: 0.0014
Epoch 6/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0019 - val_loss: 0.0020
Epoch 7/100
15/15 [==============================] - 1s 88ms/step - loss: 0.0019 - val_loss: 0.0014
Epoch 8/100
15/15 [==============================] - 2s 119ms/step - loss: 0.0018 - val_loss: 0.0017
Epoch 9/100
15/15 [==============================] - 1s 83ms/step - loss: 0.0016 - val_loss: 0.0016
Epoch 10/100
15/15 [==============================] - 1s 80ms/step - loss: 0.0015 - val_loss: 0.0027
Epoch 11/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0021 - val_loss: 0.0023
Epoch 12/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0015 - val_loss: 0.0017
Epoch 13/100
15/15 [==============================] - 1s 88ms/step - loss: 0.0018 - val_loss: 0.0067
Epoch 14/100
15/15 [==============================] - 1s 76ms/step - loss: 0.0023 - val_loss: 0.0040
Epoch 15/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0019 - val_loss: 0.0015
Epoch 16/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0015 - val_loss: 0.0013
Epoch 17/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0012 - val_loss: 0.0054
Epoch 18/100
15/15 [==============================] - 1s 90ms/step - loss: 0.0015 - val_loss: 0.0014
Epoch 19/100
15/15 [==============================] - 1s 69ms/step - loss: 0.0015 - val_loss: 0.0016
Epoch 20/100
15/15 [==============================] - 1s 81ms/step - loss: 0.0012 - val_loss: 0.0012
Epoch 21/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0012 - val_loss: 0.0025
Epoch 22/100
15/15 [==============================] - 1s 85ms/step - loss: 0.0015 - val_loss: 0.0052
Epoch 23/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0021 - val_loss: 0.0151
Epoch 24/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0027 - val_loss: 0.0123
Epoch 25/100
15/15 [==============================] - 1s 97ms/step - loss: 0.0018 - val_loss: 0.0027
Epoch 26/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0011 - val_loss: 0.0013
Epoch 27/100
15/15 [==============================] - 1s 93ms/step - loss: 0.0011 - val_loss: 0.0013
Epoch 28/100
15/15 [==============================] - 1s 75ms/step - loss: 9.5961e-04 - val_loss: 0.0029
Epoch 29/100
15/15 [==============================] - 1s 82ms/step - loss: 0.0011 - val_loss: 0.0011
Epoch 30/100
15/15 [==============================] - 1s 92ms/step - loss: 0.0013 - val_loss: 0.0013
Epoch 31/100
15/15 [==============================] - 2s 145ms/step - loss: 0.0010 - val_loss: 0.0021
Epoch 32/100
15/15 [==============================] - 1s 89ms/step - loss: 9.9007e-04 - val_loss: 0.0028
Epoch 33/100
15/15 [==============================] - 2s 115ms/step - loss: 0.0010 - val_loss: 8.8229e-04
Epoch 34/100
15/15 [==============================] - 2s 118ms/step - loss: 0.0010 - val_loss: 0.0014
Epoch 35/100
15/15 [==============================] - 2s 135ms/step - loss: 0.0011 - val_loss: 0.0091
Epoch 36/100
15/15 [==============================] - 1s 87ms/step - loss: 0.0022 - val_loss: 9.6719e-04
Epoch 37/100
15/15 [==============================] - 1s 82ms/step - loss: 0.0015 - val_loss: 0.0012
Epoch 38/100
15/15 [==============================] - 1s 83ms/step - loss: 9.0714e-04 - val_loss: 0.0011
Epoch 39/100
15/15 [==============================] - 1s 80ms/step - loss: 8.1228e-04 - val_loss: 0.0084
Epoch 40/100
15/15 [==============================] - 1s 87ms/step - loss: 0.0024 - val_loss: 0.0078
Epoch 41/100
15/15 [==============================] - 1s 86ms/step - loss: 0.0015 - val_loss: 0.0060
Epoch 42/100
15/15 [==============================] - 1s 86ms/step - loss: 0.0016 - val_loss: 0.0013
Epoch 43/100
15/15 [==============================] - 1s 94ms/step - loss: 8.5703e-04 - val_loss: 0.0026
Epoch 1/100
15/15 [==============================] - 13s 289ms/step - loss: 0.0850 - val_loss: 0.0626
Epoch 2/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0085 - val_loss: 0.0029
Epoch 3/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0037 - val_loss: 0.0024
Epoch 4/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0018 - val_loss: 0.0017
Epoch 5/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0017 - val_loss: 0.0026
Epoch 6/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0020 - val_loss: 0.0014
Epoch 7/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0016 - val_loss: 0.0015
Epoch 8/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0015 - val_loss: 0.0015
Epoch 9/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0015 - val_loss: 0.0015
Epoch 10/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0013 - val_loss: 0.0032
Epoch 11/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0016 - val_loss: 0.0035
Epoch 12/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0020 - val_loss: 0.0024
Epoch 13/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0020 - val_loss: 0.0100
Epoch 14/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0026 - val_loss: 0.0014
Epoch 15/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0017 - val_loss: 0.0020
Epoch 16/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0012 - val_loss: 0.0030
![png](tweet_stock_files/tweet_stock_137_1.png)

29/29 [==============================] - 1s 14ms/step
29/29 [==============================] - 1s 11ms/step
![png](tweet_stock_files/tweet_stock_137_3.png)

Root Mean Squared Error (RMSE) for Open = 8.410034872401216
Root Mean Squared Error (RMSE) for Close = 8.936666960433204
Total Root Mean Squared Error (RMSE) 8.677347028807336
![png](tweet_stock_files/tweet_stock_137_5.png)

Root Mean Squared Error (RMSE) for Open = 6.041314263852169
Root Mean Squared Error (RMSE) for Close = 6.281422627941314
Total Root Mean Squared Error (RMSE) 6.162537961969759
4/4 [==============================] - 0s 12ms/step
4/4 [==============================] - 0s 14ms/step
![png](tweet_stock_files/tweet_stock_137_7.png)

Root Mean Squared Error (RMSE) for Open = 16.909530369947653
Root Mean Squared Error (RMSE) for Close = 20.25355033450062
Total Root Mean Squared Error (RMSE) 18.656614356366113
![png](tweet_stock_files/tweet_stock_137_9.png)

Root Mean Squared Error (RMSE) for Open = 12.554480039236774
Root Mean Squared Error (RMSE) for Close = 13.89917268152197
Total Root Mean Squared Error (RMSE) 13.243903697293355
8/8 [==============================] - 0s 13ms/step
8/8 [==============================] - 0s 12ms/step
![png](tweet_stock_files/tweet_stock_137_11.png)

Root Mean Squared Error (RMSE) for Open = 25.746557147229932
Root Mean Squared Error (RMSE) for Close = 30.512252258599506
Total Root Mean Squared Error (RMSE) 28.23015004235708
![png](tweet_stock_files/tweet_stock_137_13.png)

Root Mean Squared Error (RMSE) for Open = 19.10214117784766
Root Mean Squared Error (RMSE) for Close = 21.703524438107063
Total Root Mean Squared Error (RMSE) 20.444250666311223

## 9.4 GOOGL t-1

```python
tweet_stock_pipline(df_for_training, 'GOOGL')
```
TrainX shape = (1250, 7, 14).
TrainY shape = (1250, 1, 2).
Model: "sequential_94"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_182 (Conv1D)         (None, 6, 128)            2944

max_pooling1d_182 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_183 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_183 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_470 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_470 (Dropout)       (None, 1, 2048)           0

bidirectional_471 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_471 (Dropout)       (None, 1, 1024)           0

bidirectional_472 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_472 (Dropout)       (None, 1, 512)            0

bidirectional_473 (Bidirec  (None, 1, 256)            656384
tional)

dropout_473 (Dropout)       (None, 1, 256)            0

bidirectional_474 (Bidirec  (None, 1, 128)            164352
tional)

dropout_474 (Dropout)       (None, 1, 128)            0

dense_188 (Dense)           (None, 1, 32)             4128

dense_189 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22878754 (87.28 MB)
Trainable params: 22878754 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Model: "sequential_95"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_184 (Conv1D)         (None, 6, 128)            3712

max_pooling1d_184 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_185 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_185 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_475 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_475 (Dropout)       (None, 1, 2048)           0

bidirectional_476 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_476 (Dropout)       (None, 1, 1024)           0

bidirectional_477 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_477 (Dropout)       (None, 1, 512)            0

bidirectional_478 (Bidirec  (None, 1, 256)            656384
tional)

dropout_478 (Dropout)       (None, 1, 256)            0

bidirectional_479 (Bidirec  (None, 1, 128)            164352
tional)

dropout_479 (Dropout)       (None, 1, 128)            0

dense_190 (Dense)           (None, 1, 32)             4128

dense_191 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22879522 (87.28 MB)
Trainable params: 22879522 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Epoch 1/100
15/15 [==============================] - 11s 246ms/step - loss: 0.1135 - val_loss: 0.0858
Epoch 2/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0109 - val_loss: 0.0054
Epoch 3/100
15/15 [==============================] - 1s 83ms/step - loss: 0.0030 - val_loss: 0.0113
Epoch 4/100
15/15 [==============================] - 2s 136ms/step - loss: 0.0037 - val_loss: 0.0038
Epoch 5/100
15/15 [==============================] - 1s 89ms/step - loss: 0.0035 - val_loss: 0.0065
Epoch 6/100
15/15 [==============================] - 1s 99ms/step - loss: 0.0026 - val_loss: 0.0019
Epoch 7/100
15/15 [==============================] - 2s 114ms/step - loss: 0.0018 - val_loss: 0.0037
Epoch 8/100
15/15 [==============================] - 1s 91ms/step - loss: 0.0019 - val_loss: 0.0030
Epoch 9/100
15/15 [==============================] - 1s 92ms/step - loss: 0.0019 - val_loss: 0.0016
Epoch 10/100
15/15 [==============================] - 1s 95ms/step - loss: 0.0014 - val_loss: 0.0016
Epoch 11/100
15/15 [==============================] - 2s 103ms/step - loss: 0.0018 - val_loss: 0.0041
Epoch 12/100
15/15 [==============================] - 2s 113ms/step - loss: 0.0021 - val_loss: 0.0024
Epoch 13/100
15/15 [==============================] - 2s 104ms/step - loss: 0.0015 - val_loss: 0.0017
Epoch 14/100
15/15 [==============================] - 1s 88ms/step - loss: 0.0015 - val_loss: 0.0049
Epoch 15/100
15/15 [==============================] - 2s 101ms/step - loss: 0.0023 - val_loss: 0.0039
Epoch 16/100
15/15 [==============================] - 2s 103ms/step - loss: 0.0016 - val_loss: 0.0024
Epoch 17/100
15/15 [==============================] - 2s 115ms/step - loss: 0.0017 - val_loss: 0.0013
Epoch 18/100
15/15 [==============================] - 2s 105ms/step - loss: 0.0012 - val_loss: 0.0013
Epoch 19/100
15/15 [==============================] - 1s 93ms/step - loss: 0.0014 - val_loss: 0.0013
Epoch 20/100
15/15 [==============================] - 1s 86ms/step - loss: 0.0016 - val_loss: 0.0016
Epoch 21/100
15/15 [==============================] - 1s 96ms/step - loss: 0.0018 - val_loss: 0.0016
Epoch 22/100
15/15 [==============================] - 1s 94ms/step - loss: 0.0012 - val_loss: 0.0012
Epoch 23/100
15/15 [==============================] - 1s 87ms/step - loss: 0.0012 - val_loss: 0.0011
Epoch 24/100
15/15 [==============================] - 1s 84ms/step - loss: 0.0010 - val_loss: 0.0011
Epoch 25/100
15/15 [==============================] - 1s 81ms/step - loss: 0.0013 - val_loss: 0.0037
Epoch 26/100
15/15 [==============================] - 1s 84ms/step - loss: 0.0012 - val_loss: 0.0011
Epoch 27/100
15/15 [==============================] - 1s 83ms/step - loss: 0.0012 - val_loss: 0.0033
Epoch 28/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0014 - val_loss: 0.0016
Epoch 29/100
15/15 [==============================] - 2s 114ms/step - loss: 0.0012 - val_loss: 0.0033
Epoch 30/100
15/15 [==============================] - 2s 115ms/step - loss: 0.0011 - val_loss: 0.0020
Epoch 31/100
15/15 [==============================] - 1s 85ms/step - loss: 0.0011 - val_loss: 8.3902e-04
Epoch 32/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0011 - val_loss: 0.0017
Epoch 33/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0015 - val_loss: 0.0078
Epoch 34/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0029 - val_loss: 0.0054
Epoch 35/100
15/15 [==============================] - 1s 88ms/step - loss: 0.0014 - val_loss: 0.0037
Epoch 36/100
15/15 [==============================] - 2s 103ms/step - loss: 0.0011 - val_loss: 0.0011
Epoch 37/100
15/15 [==============================] - 1s 80ms/step - loss: 0.0010 - val_loss: 9.6466e-04
Epoch 38/100
15/15 [==============================] - 1s 83ms/step - loss: 9.9070e-04 - val_loss: 0.0085
Epoch 39/100
15/15 [==============================] - 1s 84ms/step - loss: 0.0014 - val_loss: 0.0011
Epoch 40/100
15/15 [==============================] - 1s 83ms/step - loss: 0.0011 - val_loss: 9.7782e-04
Epoch 41/100
15/15 [==============================] - 1s 82ms/step - loss: 9.0903e-04 - val_loss: 8.8062e-04
Epoch 1/100
15/15 [==============================] - 10s 226ms/step - loss: 0.1123 - val_loss: 0.1398
Epoch 2/100
15/15 [==============================] - 1s 83ms/step - loss: 0.0176 - val_loss: 0.0074
Epoch 3/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0086 - val_loss: 0.0079
Epoch 4/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0039 - val_loss: 0.0028
Epoch 5/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0026 - val_loss: 0.0022
Epoch 6/100
15/15 [==============================] - 1s 87ms/step - loss: 0.0022 - val_loss: 0.0022
Epoch 7/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0021 - val_loss: 0.0028
Epoch 8/100
15/15 [==============================] - 1s 89ms/step - loss: 0.0021 - val_loss: 0.0023
Epoch 9/100
15/15 [==============================] - 1s 83ms/step - loss: 0.0019 - val_loss: 0.0021
Epoch 10/100
15/15 [==============================] - 1s 82ms/step - loss: 0.0017 - val_loss: 0.0023
Epoch 11/100
15/15 [==============================] - 2s 106ms/step - loss: 0.0018 - val_loss: 0.0020
Epoch 12/100
15/15 [==============================] - 1s 98ms/step - loss: 0.0015 - val_loss: 0.0020
Epoch 13/100
15/15 [==============================] - 1s 92ms/step - loss: 0.0016 - val_loss: 0.0017
Epoch 14/100
15/15 [==============================] - 1s 90ms/step - loss: 0.0016 - val_loss: 0.0026
Epoch 15/100
15/15 [==============================] - 2s 103ms/step - loss: 0.0021 - val_loss: 0.0044
Epoch 16/100
15/15 [==============================] - 1s 100ms/step - loss: 0.0020 - val_loss: 0.0025
Epoch 17/100
15/15 [==============================] - 2s 102ms/step - loss: 0.0020 - val_loss: 0.0044
Epoch 18/100
15/15 [==============================] - 2s 109ms/step - loss: 0.0016 - val_loss: 0.0017
Epoch 19/100
15/15 [==============================] - 2s 123ms/step - loss: 0.0014 - val_loss: 0.0014
Epoch 20/100
15/15 [==============================] - 2s 117ms/step - loss: 0.0015 - val_loss: 0.0021
Epoch 21/100
15/15 [==============================] - 2s 131ms/step - loss: 0.0017 - val_loss: 0.0029
Epoch 22/100
15/15 [==============================] - 2s 118ms/step - loss: 0.0020 - val_loss: 0.0015
Epoch 23/100
15/15 [==============================] - 2s 154ms/step - loss: 0.0017 - val_loss: 0.0028
Epoch 24/100
15/15 [==============================] - 2s 121ms/step - loss: 0.0017 - val_loss: 0.0015
Epoch 25/100
15/15 [==============================] - 2s 105ms/step - loss: 0.0011 - val_loss: 0.0016
Epoch 26/100
15/15 [==============================] - 2s 135ms/step - loss: 0.0012 - val_loss: 0.0013
Epoch 27/100
15/15 [==============================] - 2s 131ms/step - loss: 0.0011 - val_loss: 0.0011
Epoch 28/100
15/15 [==============================] - 2s 130ms/step - loss: 0.0013 - val_loss: 0.0012
Epoch 29/100
15/15 [==============================] - 2s 111ms/step - loss: 0.0011 - val_loss: 0.0011
Epoch 30/100
15/15 [==============================] - 2s 111ms/step - loss: 0.0010 - val_loss: 0.0016
Epoch 31/100
15/15 [==============================] - 2s 105ms/step - loss: 0.0012 - val_loss: 9.7103e-04
Epoch 32/100
15/15 [==============================] - 2s 111ms/step - loss: 0.0010 - val_loss: 0.0017
Epoch 33/100
15/15 [==============================] - 2s 140ms/step - loss: 0.0014 - val_loss: 0.0048
Epoch 34/100
15/15 [==============================] - 2s 105ms/step - loss: 0.0019 - val_loss: 0.0022
Epoch 35/100
15/15 [==============================] - 1s 101ms/step - loss: 0.0010 - val_loss: 0.0014
Epoch 36/100
15/15 [==============================] - 2s 102ms/step - loss: 0.0010 - val_loss: 0.0013
Epoch 37/100
15/15 [==============================] - 1s 94ms/step - loss: 0.0015 - val_loss: 0.0012
Epoch 38/100
15/15 [==============================] - 1s 84ms/step - loss: 0.0013 - val_loss: 0.0029
Epoch 39/100
15/15 [==============================] - 1s 81ms/step - loss: 0.0013 - val_loss: 0.0025
Epoch 40/100
15/15 [==============================] - 1s 91ms/step - loss: 9.3949e-04 - val_loss: 0.0011
Epoch 41/100
15/15 [==============================] - 1s 93ms/step - loss: 9.7068e-04 - val_loss: 0.0013
![png](tweet_stock_files/tweet_stock_139_1.png)

29/29 [==============================] - 2s 13ms/step
29/29 [==============================] - 1s 13ms/step
![png](tweet_stock_files/tweet_stock_139_3.png)

Root Mean Squared Error (RMSE) for Open = 9.002663061638545
Root Mean Squared Error (RMSE) for Close = 9.549155457343732
Total Root Mean Squared Error (RMSE) 9.279932977934397
![png](tweet_stock_files/tweet_stock_139_5.png)

Root Mean Squared Error (RMSE) for Open = 7.290595628237805
Root Mean Squared Error (RMSE) for Close = 7.2026512709454265
Total Root Mean Squared Error (RMSE) 7.246756858944972
4/4 [==============================] - 0s 19ms/step
4/4 [==============================] - 0s 12ms/step
![png](tweet_stock_files/tweet_stock_139_7.png)

Root Mean Squared Error (RMSE) for Open = 17.947477730809446
Root Mean Squared Error (RMSE) for Close = 21.18802390402088
Total Root Mean Squared Error (RMSE) 19.63471815248771
![png](tweet_stock_files/tweet_stock_139_9.png)

Root Mean Squared Error (RMSE) for Open = 12.713289430328333
Root Mean Squared Error (RMSE) for Close = 13.804969571158718
Total Root Mean Squared Error (RMSE) 13.270360074239065
8/8 [==============================] - 0s 12ms/step
8/8 [==============================] - 0s 13ms/step
![png](tweet_stock_files/tweet_stock_139_11.png)

Root Mean Squared Error (RMSE) for Open = 26.021401958783326
Root Mean Squared Error (RMSE) for Close = 30.64700183761725
Total Root Mean Squared Error (RMSE) 28.428437184758224
![png](tweet_stock_files/tweet_stock_139_13.png)

Root Mean Squared Error (RMSE) for Open = 18.752863745758134
Root Mean Squared Error (RMSE) for Close = 21.192556786377768
Total Root Mean Squared Error (RMSE) 20.009927058972703

## 9.5 AMZN t-1

```python
tweet_stock_pipline(df_for_training, 'AMZN')
```
TrainX shape = (1250, 7, 14).
TrainY shape = (1250, 1, 2).
Model: "sequential_96"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_186 (Conv1D)         (None, 6, 128)            2944

max_pooling1d_186 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_187 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_187 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_480 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_480 (Dropout)       (None, 1, 2048)           0

bidirectional_481 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_481 (Dropout)       (None, 1, 1024)           0

bidirectional_482 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_482 (Dropout)       (None, 1, 512)            0

bidirectional_483 (Bidirec  (None, 1, 256)            656384
tional)

dropout_483 (Dropout)       (None, 1, 256)            0

bidirectional_484 (Bidirec  (None, 1, 128)            164352
tional)

dropout_484 (Dropout)       (None, 1, 128)            0

dense_192 (Dense)           (None, 1, 32)             4128

dense_193 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22878754 (87.28 MB)
Trainable params: 22878754 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Model: "sequential_97"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_188 (Conv1D)         (None, 6, 128)            3712

max_pooling1d_188 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_189 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_189 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_485 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_485 (Dropout)       (None, 1, 2048)           0

bidirectional_486 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_486 (Dropout)       (None, 1, 1024)           0

bidirectional_487 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_487 (Dropout)       (None, 1, 512)            0

bidirectional_488 (Bidirec  (None, 1, 256)            656384
tional)

dropout_488 (Dropout)       (None, 1, 256)            0

bidirectional_489 (Bidirec  (None, 1, 128)            164352
tional)

dropout_489 (Dropout)       (None, 1, 128)            0

dense_194 (Dense)           (None, 1, 32)             4128

dense_195 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22879522 (87.28 MB)
Trainable params: 22879522 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Epoch 1/100
15/15 [==============================] - 10s 215ms/step - loss: 0.0937 - val_loss: 0.1977
Epoch 2/100
15/15 [==============================] - 1s 65ms/step - loss: 0.0162 - val_loss: 0.0212
Epoch 3/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0079 - val_loss: 0.0045
Epoch 4/100
15/15 [==============================] - 1s 67ms/step - loss: 0.0023 - val_loss: 0.0067
Epoch 5/100
15/15 [==============================] - 1s 68ms/step - loss: 0.0016 - val_loss: 0.0041
Epoch 6/100
15/15 [==============================] - 1s 69ms/step - loss: 0.0014 - val_loss: 0.0037
Epoch 7/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0014 - val_loss: 0.0051
Epoch 8/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0016 - val_loss: 0.0035
Epoch 9/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0011 - val_loss: 0.0043
Epoch 10/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0011 - val_loss: 0.0033
Epoch 11/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0013 - val_loss: 0.0037
Epoch 12/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0011 - val_loss: 0.0045
Epoch 13/100
15/15 [==============================] - 1s 69ms/step - loss: 0.0012 - val_loss: 0.0080
Epoch 14/100
15/15 [==============================] - 1s 69ms/step - loss: 0.0012 - val_loss: 0.0044
Epoch 15/100
15/15 [==============================] - 1s 69ms/step - loss: 0.0018 - val_loss: 0.0055
Epoch 16/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0016 - val_loss: 0.0081
Epoch 17/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0016 - val_loss: 0.0044
Epoch 18/100
15/15 [==============================] - 1s 74ms/step - loss: 9.7237e-04 - val_loss: 0.0025
Epoch 19/100
15/15 [==============================] - 1s 71ms/step - loss: 9.2979e-04 - val_loss: 0.0027
Epoch 20/100
15/15 [==============================] - 2s 102ms/step - loss: 0.0010 - val_loss: 0.0027
Epoch 21/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0011 - val_loss: 0.0181
Epoch 22/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0035 - val_loss: 0.0116
Epoch 23/100
15/15 [==============================] - 1s 87ms/step - loss: 0.0022 - val_loss: 0.0036
Epoch 24/100
15/15 [==============================] - 1s 101ms/step - loss: 0.0016 - val_loss: 0.0035
Epoch 25/100
15/15 [==============================] - 1s 79ms/step - loss: 8.9323e-04 - val_loss: 0.0045
Epoch 26/100
15/15 [==============================] - 1s 79ms/step - loss: 7.5731e-04 - val_loss: 0.0033
Epoch 27/100
15/15 [==============================] - 1s 90ms/step - loss: 7.5665e-04 - val_loss: 0.0029
Epoch 28/100
15/15 [==============================] - 1s 81ms/step - loss: 9.2090e-04 - val_loss: 0.0027
Epoch 1/100
15/15 [==============================] - 11s 267ms/step - loss: 0.1228 - val_loss: 0.4688
Epoch 2/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0834 - val_loss: 0.3732
Epoch 3/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0786 - val_loss: 0.3706
Epoch 4/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0783 - val_loss: 0.3646
Epoch 5/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0782 - val_loss: 0.3647
Epoch 6/100
15/15 [==============================] - 2s 107ms/step - loss: 0.0629 - val_loss: 0.0636
Epoch 7/100
15/15 [==============================] - 2s 107ms/step - loss: 0.0051 - val_loss: 0.0050
Epoch 8/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0021 - val_loss: 0.0045
Epoch 9/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0013 - val_loss: 0.0091
Epoch 10/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0019 - val_loss: 0.0037
Epoch 11/100
15/15 [==============================] - 1s 89ms/step - loss: 0.0021 - val_loss: 0.0041
Epoch 12/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0013 - val_loss: 0.0069
Epoch 13/100
15/15 [==============================] - 1s 80ms/step - loss: 9.7995e-04 - val_loss: 0.0041
Epoch 14/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0010 - val_loss: 0.0049
Epoch 15/100
15/15 [==============================] - 1s 76ms/step - loss: 0.0015 - val_loss: 0.0105
Epoch 16/100
15/15 [==============================] - 1s 87ms/step - loss: 0.0014 - val_loss: 0.0093
Epoch 17/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0013 - val_loss: 0.0037
Epoch 18/100
15/15 [==============================] - 1s 76ms/step - loss: 9.0635e-04 - val_loss: 0.0036
Epoch 19/100
15/15 [==============================] - 1s 82ms/step - loss: 8.9658e-04 - val_loss: 0.0029
Epoch 20/100
15/15 [==============================] - 1s 77ms/step - loss: 8.9702e-04 - val_loss: 0.0042
Epoch 21/100
15/15 [==============================] - 1s 85ms/step - loss: 9.3953e-04 - val_loss: 0.0187
Epoch 22/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0032 - val_loss: 0.0086
Epoch 23/100
15/15 [==============================] - 1s 76ms/step - loss: 0.0019 - val_loss: 0.0045
Epoch 24/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0011 - val_loss: 0.0066
Epoch 25/100
15/15 [==============================] - 1s 81ms/step - loss: 9.9099e-04 - val_loss: 0.0044
Epoch 26/100
15/15 [==============================] - 1s 86ms/step - loss: 7.7034e-04 - val_loss: 0.0033
Epoch 27/100
15/15 [==============================] - 1s 84ms/step - loss: 8.5342e-04 - val_loss: 0.0042
Epoch 28/100
15/15 [==============================] - 1s 82ms/step - loss: 7.9109e-04 - val_loss: 0.0067
Epoch 29/100
15/15 [==============================] - 1s 85ms/step - loss: 0.0010 - val_loss: 0.0044
![png](tweet_stock_files/tweet_stock_141_1.png)

29/29 [==============================] - 2s 17ms/step
29/29 [==============================] - 1s 13ms/step
![png](tweet_stock_files/tweet_stock_141_3.png)

Root Mean Squared Error (RMSE) for Open = 9.983211578026202
Root Mean Squared Error (RMSE) for Close = 10.066535978055557
Total Root Mean Squared Error (RMSE) 10.024960349276286
![png](tweet_stock_files/tweet_stock_141_5.png)

Root Mean Squared Error (RMSE) for Open = 8.568611810983436
Root Mean Squared Error (RMSE) for Close = 8.3735129167291
Total Root Mean Squared Error (RMSE) 8.471624015911999
4/4 [==============================] - 0s 12ms/step
4/4 [==============================] - 0s 12ms/step
![png](tweet_stock_files/tweet_stock_141_7.png)

Root Mean Squared Error (RMSE) for Open = 31.785410993910684
Root Mean Squared Error (RMSE) for Close = 35.2659935654329
Total Root Mean Squared Error (RMSE) 33.5708404289271
![png](tweet_stock_files/tweet_stock_141_9.png)

Root Mean Squared Error (RMSE) for Open = 23.343435129238422
Root Mean Squared Error (RMSE) for Close = 24.74154105818535
Total Root Mean Squared Error (RMSE) 24.052648685402964
8/8 [==============================] - 0s 11ms/step
8/8 [==============================] - 0s 12ms/step
![png](tweet_stock_files/tweet_stock_141_11.png)

Root Mean Squared Error (RMSE) for Open = 30.216223392220012
Root Mean Squared Error (RMSE) for Close = 34.61686613045786
Total Root Mean Squared Error (RMSE) 32.49113399669658
![png](tweet_stock_files/tweet_stock_141_13.png)

Root Mean Squared Error (RMSE) for Open = 20.616232027318386
Root Mean Squared Error (RMSE) for Close = 22.764266198015243
Total Root Mean Squared Error (RMSE) 21.71682341571078

## 9.6 TSLA t-1

```python
tweet_stock_pipline(df_for_training, 'TSLA')
```
TrainX shape = (1250, 7, 14).
TrainY shape = (1250, 1, 2).
Model: "sequential_98"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_190 (Conv1D)         (None, 6, 128)            2944

max_pooling1d_190 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_191 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_191 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_490 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_490 (Dropout)       (None, 1, 2048)           0

bidirectional_491 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_491 (Dropout)       (None, 1, 1024)           0

bidirectional_492 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_492 (Dropout)       (None, 1, 512)            0

bidirectional_493 (Bidirec  (None, 1, 256)            656384
tional)

dropout_493 (Dropout)       (None, 1, 256)            0

bidirectional_494 (Bidirec  (None, 1, 128)            164352
tional)

dropout_494 (Dropout)       (None, 1, 128)            0

dense_196 (Dense)           (None, 1, 32)             4128

dense_197 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22878754 (87.28 MB)
Trainable params: 22878754 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Model: "sequential_99"

---

Layer (type)                Output Shape              Param #
=============================================================

conv1d_192 (Conv1D)         (None, 6, 128)            3712

max_pooling1d_192 (MaxPool  (None, 3, 128)            0
ing1D)

conv1d_193 (Conv1D)         (None, 2, 64)             16448

max_pooling1d_193 (MaxPool  (None, 1, 64)             0
ing1D)

bidirectional_495 (Bidirec  (None, 1, 2048)           8921088
tional)

dropout_495 (Dropout)       (None, 1, 2048)           0

bidirectional_496 (Bidirec  (None, 1, 1024)           10489856
tional)

dropout_496 (Dropout)       (None, 1, 1024)           0

bidirectional_497 (Bidirec  (None, 1, 512)            2623488
tional)

dropout_497 (Dropout)       (None, 1, 512)            0

bidirectional_498 (Bidirec  (None, 1, 256)            656384
tional)

dropout_498 (Dropout)       (None, 1, 256)            0

bidirectional_499 (Bidirec  (None, 1, 128)            164352
tional)

dropout_499 (Dropout)       (None, 1, 128)            0

dense_198 (Dense)           (None, 1, 32)             4128

dense_199 (Dense)           (None, 1, 2)              66

=================================================================
Total params: 22879522 (87.28 MB)
Trainable params: 22879522 (87.28 MB)
Non-trainable params: 0 (0.00 Byte)

---

Epoch 1/100
15/15 [==============================] - 10s 237ms/step - loss: 0.1077 - val_loss: 0.0727
Epoch 2/100
15/15 [==============================] - 1s 64ms/step - loss: 0.0201 - val_loss: 0.0253
Epoch 3/100
15/15 [==============================] - 1s 69ms/step - loss: 0.0097 - val_loss: 0.0099
Epoch 4/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0054 - val_loss: 0.0045
Epoch 5/100
15/15 [==============================] - 1s 76ms/step - loss: 0.0033 - val_loss: 0.0051
Epoch 6/100
15/15 [==============================] - 1s 90ms/step - loss: 0.0030 - val_loss: 0.0037
Epoch 7/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0033 - val_loss: 0.0038
Epoch 8/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0027 - val_loss: 0.0033
Epoch 9/100
15/15 [==============================] - 1s 83ms/step - loss: 0.0026 - val_loss: 0.0031
Epoch 10/100
15/15 [==============================] - 1s 82ms/step - loss: 0.0025 - val_loss: 0.0110
Epoch 11/100
15/15 [==============================] - 1s 85ms/step - loss: 0.0032 - val_loss: 0.0115
Epoch 12/100
15/15 [==============================] - 1s 85ms/step - loss: 0.0033 - val_loss: 0.0047
Epoch 13/100
15/15 [==============================] - 1s 84ms/step - loss: 0.0021 - val_loss: 0.0040
Epoch 14/100
15/15 [==============================] - 2s 105ms/step - loss: 0.0021 - val_loss: 0.0141
Epoch 15/100
15/15 [==============================] - 2s 109ms/step - loss: 0.0038 - val_loss: 0.0118
Epoch 16/100
15/15 [==============================] - 1s 88ms/step - loss: 0.0031 - val_loss: 0.0034
Epoch 17/100
15/15 [==============================] - 1s 101ms/step - loss: 0.0021 - val_loss: 0.0035
Epoch 18/100
15/15 [==============================] - 1s 89ms/step - loss: 0.0019 - val_loss: 0.0057
Epoch 19/100
15/15 [==============================] - 2s 104ms/step - loss: 0.0020 - val_loss: 0.0023
Epoch 20/100
15/15 [==============================] - 2s 100ms/step - loss: 0.0020 - val_loss: 0.0033
Epoch 21/100
15/15 [==============================] - 2s 102ms/step - loss: 0.0019 - val_loss: 0.0039
Epoch 22/100
15/15 [==============================] - 1s 100ms/step - loss: 0.0021 - val_loss: 0.0021
Epoch 23/100
15/15 [==============================] - 2s 102ms/step - loss: 0.0015 - val_loss: 0.0016
Epoch 24/100
15/15 [==============================] - 2s 110ms/step - loss: 0.0014 - val_loss: 0.0016
Epoch 25/100
15/15 [==============================] - 2s 142ms/step - loss: 0.0018 - val_loss: 0.0017
Epoch 26/100
15/15 [==============================] - 2s 106ms/step - loss: 0.0016 - val_loss: 0.0016
Epoch 27/100
15/15 [==============================] - 1s 88ms/step - loss: 0.0012 - val_loss: 0.0015
Epoch 28/100
15/15 [==============================] - 1s 87ms/step - loss: 0.0014 - val_loss: 0.0027
Epoch 29/100
15/15 [==============================] - 1s 81ms/step - loss: 0.0013 - val_loss: 0.0026
Epoch 30/100
15/15 [==============================] - 1s 85ms/step - loss: 0.0013 - val_loss: 0.0069
Epoch 31/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0019 - val_loss: 0.0046
Epoch 32/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0016 - val_loss: 0.0045
Epoch 33/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0015 - val_loss: 0.0016
Epoch 34/100
15/15 [==============================] - 1s 83ms/step - loss: 0.0012 - val_loss: 0.0018
Epoch 35/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0011 - val_loss: 0.0018
Epoch 36/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0011 - val_loss: 0.0015
Epoch 37/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0012 - val_loss: 0.0018
Epoch 38/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0013 - val_loss: 0.0022
Epoch 39/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0010 - val_loss: 0.0017
Epoch 40/100
15/15 [==============================] - 1s 73ms/step - loss: 9.5484e-04 - val_loss: 0.0016
Epoch 41/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0012 - val_loss: 0.0015
Epoch 42/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0012 - val_loss: 0.0025
Epoch 43/100
15/15 [==============================] - 1s 73ms/step - loss: 8.5746e-04 - val_loss: 0.0026
Epoch 44/100
15/15 [==============================] - 1s 95ms/step - loss: 0.0011 - val_loss: 0.0025
Epoch 45/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0019 - val_loss: 0.0019
Epoch 46/100
15/15 [==============================] - 2s 128ms/step - loss: 0.0012 - val_loss: 0.0016
Epoch 1/100
15/15 [==============================] - 11s 208ms/step - loss: 0.1152 - val_loss: 0.0717
Epoch 2/100
15/15 [==============================] - 1s 66ms/step - loss: 0.0185 - val_loss: 0.0061
Epoch 3/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0082 - val_loss: 0.0092
Epoch 4/100
15/15 [==============================] - 1s 73ms/step - loss: 0.0040 - val_loss: 0.0050
Epoch 5/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0033 - val_loss: 0.0050
Epoch 6/100
15/15 [==============================] - 1s 80ms/step - loss: 0.0030 - val_loss: 0.0046
Epoch 7/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0030 - val_loss: 0.0045
Epoch 8/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0028 - val_loss: 0.0043
Epoch 9/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0028 - val_loss: 0.0046
Epoch 10/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0026 - val_loss: 0.0053
Epoch 11/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0026 - val_loss: 0.0079
Epoch 12/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0027 - val_loss: 0.0045
Epoch 13/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0022 - val_loss: 0.0032
Epoch 14/100
15/15 [==============================] - 2s 107ms/step - loss: 0.0023 - val_loss: 0.0081
Epoch 15/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0030 - val_loss: 0.0058
Epoch 16/100
15/15 [==============================] - 1s 76ms/step - loss: 0.0024 - val_loss: 0.0030
Epoch 17/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0023 - val_loss: 0.0025
Epoch 18/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0018 - val_loss: 0.0031
Epoch 19/100
15/15 [==============================] - 1s 76ms/step - loss: 0.0019 - val_loss: 0.0027
Epoch 20/100
15/15 [==============================] - 1s 70ms/step - loss: 0.0019 - val_loss: 0.0027
Epoch 21/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0019 - val_loss: 0.0061
Epoch 22/100
15/15 [==============================] - 1s 76ms/step - loss: 0.0025 - val_loss: 0.0030
Epoch 23/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0019 - val_loss: 0.0037
Epoch 24/100
15/15 [==============================] - 1s 76ms/step - loss: 0.0017 - val_loss: 0.0022
Epoch 25/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0019 - val_loss: 0.0024
Epoch 26/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0015 - val_loss: 0.0024
Epoch 27/100
15/15 [==============================] - 1s 89ms/step - loss: 0.0014 - val_loss: 0.0019
Epoch 28/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0013 - val_loss: 0.0019
Epoch 29/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0015 - val_loss: 0.0027
Epoch 30/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0015 - val_loss: 0.0029
Epoch 31/100
15/15 [==============================] - 1s 80ms/step - loss: 0.0013 - val_loss: 0.0037
Epoch 32/100
15/15 [==============================] - 1s 79ms/step - loss: 0.0014 - val_loss: 0.0036
Epoch 33/100
15/15 [==============================] - 1s 74ms/step - loss: 0.0017 - val_loss: 0.0025
Epoch 34/100
15/15 [==============================] - 1s 72ms/step - loss: 0.0013 - val_loss: 0.0054
Epoch 35/100
15/15 [==============================] - 1s 77ms/step - loss: 0.0023 - val_loss: 0.0052
Epoch 36/100
15/15 [==============================] - 1s 71ms/step - loss: 0.0015 - val_loss: 0.0038
Epoch 37/100
15/15 [==============================] - 1s 75ms/step - loss: 0.0015 - val_loss: 0.0036
![png](tweet_stock_files/tweet_stock_143_1.png)

29/29 [==============================] - 2s 19ms/step
29/29 [==============================] - 1s 12ms/step
![png](tweet_stock_files/tweet_stock_143_3.png)

Root Mean Squared Error (RMSE) for Open = 10.290773895213501
Root Mean Squared Error (RMSE) for Close = 13.08898095724422
Total Root Mean Squared Error (RMSE) 11.773305607634358
![png](tweet_stock_files/tweet_stock_143_5.png)

Root Mean Squared Error (RMSE) for Open = 12.700664499145276
Root Mean Squared Error (RMSE) for Close = 14.626838848607138
Total Root Mean Squared Error (RMSE) 13.697651138479394
4/4 [==============================] - 0s 18ms/step
4/4 [==============================] - 0s 10ms/step
![png](tweet_stock_files/tweet_stock_143_7.png)

Root Mean Squared Error (RMSE) for Open = 15.308795846566442
Root Mean Squared Error (RMSE) for Close = 18.92394500038692
Total Root Mean Squared Error (RMSE) 17.211550259193956
![png](tweet_stock_files/tweet_stock_143_9.png)

Root Mean Squared Error (RMSE) for Open = 12.676615423947682
Root Mean Squared Error (RMSE) for Close = 14.732194042993338
Total Root Mean Squared Error (RMSE) 13.742891251972358
8/8 [==============================] - 0s 12ms/step
8/8 [==============================] - 0s 12ms/step
![png](tweet_stock_files/tweet_stock_143_11.png)

Root Mean Squared Error (RMSE) for Open = 13.28698506810702
Root Mean Squared Error (RMSE) for Close = 16.611078546973967
Total Root Mean Squared Error (RMSE) 15.041141956211929
![png](tweet_stock_files/tweet_stock_143_13.png)

Root Mean Squared Error (RMSE) for Open = 10.691257613780026
Root Mean Squared Error (RMSE) for Close = 12.613881135527187
Total Root Mean Squared Error (RMSE) 11.692155204781924

```python

```
```python

```
```python

```
# 10. Binary Classification Task

## Creating Binary Labels

```python
stock_data_senti.head()
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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
      <th>ticker_symbol</th>
      <th>ma_7</th>
      <th>ma_30</th>
      <th>volatility_7</th>
      <th>total_engagement</th>
      <th>tweet_volume</th>
      <th>target</th>
      <th>textblob_polarity</th>
      <th>vader_polarity</th>
      <th>tweetnlp_polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>289.930</td>
      <td>293.680</td>
      <td>289.520</td>
      <td>293.650</td>
      <td>292.955</td>
      <td>25201400</td>
      <td>AAPL</td>
      <td>300.349</td>
      <td>313.548</td>
      <td>5.029</td>
      <td>2288.000</td>
      <td>466489.000</td>
      <td>0</td>
      <td>0.354</td>
      <td>0.754</td>
      <td>0.583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-12-30</td>
      <td>289.460</td>
      <td>292.690</td>
      <td>285.220</td>
      <td>291.520</td>
      <td>290.830</td>
      <td>36028600</td>
      <td>AAPL</td>
      <td>297.761</td>
      <td>312.359</td>
      <td>4.014</td>
      <td>3256.000</td>
      <td>535824.000</td>
      <td>0</td>
      <td>1.402</td>
      <td>1.694</td>
      <td>1.230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-12-27</td>
      <td>291.120</td>
      <td>293.970</td>
      <td>288.120</td>
      <td>289.800</td>
      <td>289.114</td>
      <td>36566500</td>
      <td>AAPL</td>
      <td>295.849</td>
      <td>311.365</td>
      <td>4.183</td>
      <td>1194.000</td>
      <td>145161.000</td>
      <td>1</td>
      <td>0.316</td>
      <td>0.604</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-26</td>
      <td>284.820</td>
      <td>289.980</td>
      <td>284.700</td>
      <td>289.910</td>
      <td>289.224</td>
      <td>23280300</td>
      <td>AAPL</td>
      <td>294.637</td>
      <td>310.311</td>
      <td>4.537</td>
      <td>2613.000</td>
      <td>477481.000</td>
      <td>0</td>
      <td>0.417</td>
      <td>0.581</td>
      <td>-0.023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-24</td>
      <td>284.690</td>
      <td>284.890</td>
      <td>282.920</td>
      <td>284.270</td>
      <td>283.597</td>
      <td>12119700</td>
      <td>AAPL</td>
      <td>292.419</td>
      <td>309.119</td>
      <td>5.321</td>
      <td>1348.000</td>
      <td>123904.000</td>
      <td>0</td>
      <td>0.812</td>
      <td>0.847</td>
      <td>-0.043</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Create labels for t+1
stock_data_senti['label_t1'] = (stock_data_senti['close'].shift(-1) > stock_data_senti['close']).astype(int)

# Create labels for t+7
stock_data_senti['label_t7'] = (stock_data_senti['close'].shift(-7) > stock_data_senti['close']).astype(int)

# Drop rows where labels are NaN due to shifting
stock_data_senti.dropna(subset=['label_t1', 'label_t7'], inplace=True)

# Display the updated stock_data_sentiset with labels
stock_data_senti[['date', 'close', 'label_t1', 'label_t7']].head()
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
      <th>date</th>
      <th>close</th>
      <th>label_t1</th>
      <th>label_t7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>293.650</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-12-30</td>
      <td>291.520</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-12-27</td>
      <td>289.800</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-26</td>
      <td>289.910</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-24</td>
      <td>284.270</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

## 1. Select Features

```python
# Selecting features
features = [
    'open', 'high', 'low', 'close', 'adj close', 'volume',
    'ma_7', 'ma_30', 'volatility_7', 'total_engagement',
    'tweet_volume', 'textblob_polarity', 'vader_polarity', 'tweetnlp_polarity'
]

X = stock_data_senti[features]
y_t1 = stock_data_senti['label_t1']
y_t7 = stock_data_senti['label_t7']
```
## 2. Split Data

```python
from sklearn.model_selection import train_test_split

# Split the data
X_train_t1, X_test_t1, y_train_t1, y_test_t1 = train_test_split(X, y_t1, test_size=0.2, random_state=42)
X_train_t7, X_test_t7, y_train_t7, y_test_t7 = train_test_split(X, y_t7, test_size=0.2, random_state=42)
```
## 3. Normalize Features

```python
from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_t1_scaled = scaler.fit_transform(X_train_t1)
X_test_t1_scaled = scaler.transform(X_test_t1)
X_train_t7_scaled = scaler.fit_transform(X_train_t7)
X_test_t7_scaled = scaler.transform(X_test_t7)
```
## 4. Train Models

### Logistic Regression for  t+1

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize the Logistic Regression model
log_reg_t1 = LogisticRegression(random_state=42)

# Train the model
log_reg_t1.fit(X_train_t1_scaled, y_train_t1)

# Make predictions
y_pred_t1 = log_reg_t1.predict(X_test_t1_scaled)

# Evaluate the model
precision_t1 = precision_score(y_test_t1, y_pred_t1)
recall_t1 = recall_score(y_test_t1, y_pred_t1)
f1_t1 = f1_score(y_test_t1, y_pred_t1)

# Display the evaluation metrics
(precision_t1, recall_t1, f1_t1)
```
(0.7214137214137214, 0.4894217207334274, 0.5831932773109244)

### Logistic Regression for  t+7

```python
# Initialize the Logistic Regression model
log_reg_t7 = LogisticRegression(random_state=42)

# Train the model
log_reg_t7.fit(X_train_t7_scaled, y_train_t7)

# Make predictions
y_pred_t7 = log_reg_t7.predict(X_test_t7_scaled)

# Evaluate the model
precision_t7 = precision_score(y_test_t7, y_pred_t7)
recall_t7 = recall_score(y_test_t7, y_pred_t7)
f1_t7 = f1_score(y_test_t7, y_pred_t7)

# Display the evaluation metrics
(precision_t7, recall_t7, f1_t7)
```
(0.6255924170616114, 0.21019108280254778, 0.3146603098927294)

### Random Forest Classifier for  t+1  and  t+7

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model for t+1
rf_t1 = RandomForestClassifier(random_state=42)

# Train the model
rf_t1.fit(X_train_t1_scaled, y_train_t1)

# Make predictions
y_pred_t1_rf = rf_t1.predict(X_test_t1_scaled)

# Evaluate the model
precision_t1_rf = precision_score(y_test_t1, y_pred_t1_rf)
recall_t1_rf = recall_score(y_test_t1, y_pred_t1_rf)
f1_t1_rf = f1_score(y_test_t1, y_pred_t1_rf)

# Initialize the Random Forest model for t+7
rf_t7 = RandomForestClassifier(random_state=42)

# Train the model
rf_t7.fit(X_train_t7_scaled, y_train_t7)

# Make predictions
y_pred_t7_rf = rf_t7.predict(X_test_t7_scaled)

# Evaluate the model
precision_t7_rf = precision_score(y_test_t7, y_pred_t7_rf)
recall_t7_rf = recall_score(y_test_t7, y_pred_t7_rf)
f1_t7_rf = f1_score(y_test_t7, y_pred_t7_rf)

(precision_t1_rf, recall_t1_rf, f1_t1_rf), (precision_t7_rf, recall_t7_rf, f1_t7_rf)
```
((0.5789473684210527, 0.49647390691114246, 0.5345482156416097),
(0.6872427983539094, 0.5318471337579618, 0.599640933572711))

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize the performance metrics
def plot_metrics(metrics, title):
    labels = ['Precision', 'Recall', 'F1 Score']
    t1_scores = [metrics[0][0], metrics[0][1], metrics[0][2]]
    t7_scores = [metrics[1][0], metrics[1][1], metrics[1][2]]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, t1_scores, width, label='t+1')
    rects2 = ax.bar(x + width/2, t7_scores, width, label='t+7')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Attach a text label above each bar in rects, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

# Metrics for Logistic Regression
log_reg_metrics = (
    (precision_t1, recall_t1, f1_t1),
    (precision_t7, recall_t7, f1_t7)
)

# Metrics for Random Forest
rf_metrics = (
    (precision_t1_rf, recall_t1_rf, f1_t1_rf),
    (precision_t7_rf, recall_t7_rf, f1_t7_rf)
)

# Plot metrics for Logistic Regression
plot_metrics(log_reg_metrics, 'Logistic Regression Performance Metrics')

# Plot metrics for Random Forest
plot_metrics(rf_metrics, 'Random Forest Performance Metrics')
```
![png](tweet_stock_files/tweet_stock_165_0.png)

![png](tweet_stock_files/tweet_stock_165_1.png)

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the GridSearchCV object for t+1
grid_search_t1 = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf,
                              cv=3, n_jobs=-1, scoring='f1', verbose=2)

# Fit the grid search to the data for t+1
grid_search_t1.fit(X_train_t1_scaled, y_train_t1)

# Get the best parameters and model for t+1
best_params_t1 = grid_search_t1.best_params_
best_model_t1 = grid_search_t1.best_estimator_

best_params_t1
```
Fitting 3 folds for each of 216 candidates, totalling 648 fits
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   4.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   4.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   4.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   4.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   4.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   4.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   4.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   4.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   4.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.0s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   3.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   3.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   3.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   4.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   4.4s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.4s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   4.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   4.4s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   4.4s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   4.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   4.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   4.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   4.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.2s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.4s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   4.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   6.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   4.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   4.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   3.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   3.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   3.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   6.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   3.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   7.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   3.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   6.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   3.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   9.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   9.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   6.3s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=  10.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   6.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   3.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=  10.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=  10.3s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   7.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   7.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   9.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=  10.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=  10.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=  10.3s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   7.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   7.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   7.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.3s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=  11.3s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=  11.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   7.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=  11.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   7.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   7.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   3.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   9.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=  10.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   9.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   6.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   9.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   8.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   7.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   4.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   7.3s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   7.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   7.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   2.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   5.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   5.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   5.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.3s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   4.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   4.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.2s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.2s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.3s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.3s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.3s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   7.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   8.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   5.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   5.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   7.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   8.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   5.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   5.3s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   7.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   5.3s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   3.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   3.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   8.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   9.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   8.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   6.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   7.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   6.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   8.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   9.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   8.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.3s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.3s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   8.3s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   8.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   5.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   5.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   7.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   4.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   4.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   4.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   6.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   6.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   6.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   7.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   7.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   7.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   5.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   8.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   8.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   8.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   3.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   3.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   3.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   9.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=  10.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   6.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   9.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   6.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   6.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   9.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   9.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   8.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   6.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   6.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   7.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   7.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   7.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   8.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   7.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   8.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   8.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   6.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   6.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   6.0s

{'bootstrap': True,
'max_depth': None,
'min_samples_leaf': 1,
'min_samples_split': 2,
'n_estimators': 300}

```python
# Initialize the GridSearchCV object for t+7
grid_search_t7 = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf,
                              cv=3, n_jobs=-1, scoring='f1', verbose=2)

# Fit the grid search to the data for t+7
grid_search_t7.fit(X_train_t7_scaled, y_train_t7)

# Get the best parameters and model for t+7
best_params_t7 = grid_search_t7.best_params_
best_model_t7 = grid_search_t7.best_estimator_

best_params_t1, best_params_t7
```
Fitting 3 folds for each of 216 candidates, totalling 648 fits
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   4.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   6.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   6.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   6.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   6.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   5.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   5.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   5.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   4.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.5s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.2s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.8s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.8s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   2.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.3s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.7s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   3.9s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   4.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   2.9s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   4.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   4.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   4.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.6s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.6s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   2.6s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   4.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   4.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.9s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   4.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   2.7s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.8s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.8s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   2.9s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   4.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   4.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   4.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.7s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.8s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   2.6s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.4s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   4.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   4.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.6s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   4.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   2.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.3s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   3.8s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   3.7s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.6s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   3.8s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   2.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.2s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.1s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.7s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.7s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   3.7s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.5s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   2.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   3.8s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   3.9s
[CV] END bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   3.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.4s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   4.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.2s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   6.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   4.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   4.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   6.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.2s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.2s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   6.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   7.0s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   4.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   6.4s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.6s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.7s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.1s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   3.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   5.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   5.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   5.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   6.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   4.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   4.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   1.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   6.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   6.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   6.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.2s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   6.2s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   6.1s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.7s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   4.0s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.6s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.3s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.2s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.9s
[CV] END bootstrap=True, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   4.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   3.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   3.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   9.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   9.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   9.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   6.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   6.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   6.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   3.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   8.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   9.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   9.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   9.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   9.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   9.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   6.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   6.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   3.3s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   9.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   9.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   7.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=  10.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   7.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   7.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   3.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   3.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   3.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=  10.3s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=  10.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   6.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=  10.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   6.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   6.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   3.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   9.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   9.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   6.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   8.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   8.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   5.6s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   6.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   9.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   8.3s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.0s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.1s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   8.3s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.3s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   8.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   4.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   8.7s
[CV] END bootstrap=False, max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   8.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   6.2s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   6.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   4.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   4.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   6.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   4.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   5.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   6.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   5.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   4.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   3.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   6.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   6.2s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   4.2s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   4.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   6.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   4.3s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   7.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   7.1s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   4.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   4.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   6.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   4.4s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.0s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   6.2s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   6.2s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   5.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   3.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   1.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   5.5s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.9s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   1.8s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   5.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   3.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   3.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   6.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   3.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   4.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=  10.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=  10.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   7.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=  10.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   7.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   3.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   7.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   3.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   3.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=  11.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=  11.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=  10.3s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   6.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   3.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   3.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   3.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   9.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   9.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   9.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   8.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   8.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   8.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   8.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   9.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   6.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   6.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=  10.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   6.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   3.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   3.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   3.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=  10.1s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=  10.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   9.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   6.2s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   6.0s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   8.3s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   8.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.6s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   8.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   7.5s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   7.8s
[CV] END bootstrap=False, max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   7.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   5.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100; total time=   3.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300; total time=   8.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   7.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   7.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=300; total time=   7.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200; total time=   5.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   7.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   7.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=300; total time=   7.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   4.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   2.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=100; total time=   2.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   8.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=300; total time=   8.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   5.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=100; total time=   2.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.1s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300; total time=   8.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=200; total time=   5.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   7.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   7.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   4.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=2, min_samples_split=10, n_estimators=300; total time=   7.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   4.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=100; total time=   2.5s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   7.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   7.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   4.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=2, n_estimators=300; total time=   7.2s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   4.7s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=200; total time=   4.8s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.4s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=100; total time=   2.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.3s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=5, n_estimators=300; total time=   7.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=200; total time=   5.0s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.9s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.6s
[CV] END bootstrap=False, max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300; total time=   5.0s

({'bootstrap': True,
'max_depth': None,
'min_samples_leaf': 1,
'min_samples_split': 2,
'n_estimators': 300},
{'bootstrap': False,
'max_depth': None,
'min_samples_leaf': 1,
'min_samples_split': 5,
'n_estimators': 200})

## Train and Evaluate Models

### t+1  Model

```python

# Train the Random Forest model with the best parameters for t+1
rf_t1_best = RandomForestClassifier(**best_params_t1, random_state=42)
rf_t1_best.fit(X_train_t1_scaled, y_train_t1)

# Make predictions
y_pred_t1_best = rf_t1_best.predict(X_test_t1_scaled)

# Evaluate the model
precision_t1_best = precision_score(y_test_t1, y_pred_t1_best)
recall_t1_best = recall_score(y_test_t1, y_pred_t1_best)
f1_t1_best = f1_score(y_test_t1, y_pred_t1_best)

(precision_t1_best, recall_t1_best, f1_t1_best)
```
(0.584518167456556, 0.5218617771509168, 0.5514157973174366)

### t+7  Model

```python
# Train the Random Forest model with the best parameters for t+7
rf_t7_best = RandomForestClassifier(**best_params_t7, random_state=42)
rf_t7_best.fit(X_train_t7_scaled, y_train_t7)

# Make predictions
y_pred_t7_best = rf_t7_best.predict(X_test_t7_scaled)

# Evaluate the model
precision_t7_best = precision_score(y_test_t7, y_pred_t7_best)
recall_t7_best = recall_score(y_test_t7, y_pred_t7_best)
f1_t7_best = f1_score(y_test_t7, y_pred_t7_best)

(precision_t7_best, recall_t7_best, f1_t7_best)
```
(0.67578125, 0.5509554140127388, 0.6070175438596491)

## Tuning Result Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Metrics before tuning
before_tuning = {
    't+1': (precision_t1_rf, recall_t1_rf, f1_t1_rf),
    't+7': (precision_t7_rf, recall_t7_rf, f1_t7_rf)
}

# Metrics after tuning
after_tuning = {
    't+1': (precision_t1_best, recall_t1_best, f1_t1_best),
    't+7': (precision_t7_best, recall_t7_best, f1_t7_best)
}

labels = ['Precision', 'Recall', 'F1 Score']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot for t+1
ax[0].bar(x - width/2, before_tuning['t+1'], width, label='Before Tuning')
ax[0].bar(x + width/2, after_tuning['t+1'], width, label='After Tuning')
ax[0].set_ylabel('Scores')
ax[0].set_title('Performance Comparison for t+1')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].legend()

# Plot for t+7
ax[1].bar(x - width/2, before_tuning['t+7'], width, label='Before Tuning')
ax[1].bar(x + width/2, after_tuning['t+7'], width, label='After Tuning')
ax[1].set_ylabel('Scores')
ax[1].set_title('Performance Comparison for t+7')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].legend()

fig.tight_layout()
plt.show()
```
![png](tweet_stock_files/tweet_stock_175_0.png)

## Market simulation

To show the economic value of your model, you can simulate real market trading. The basic idea
is to mimic the behavior of a trader who make decision based on your model. The trader has
$10,000 as seed money. If the model forecasts that an individual stock price will move up the
next day, the trader will invest in $10,000 worth of that stock at the opening price and sells the
stock at the closing price. If the model forecasts that an individual stock price will move down,
the trader will take no action. You can calculate the average profits over the individual stocks that
are presented in the dataset, e.g., average profit on day t+1 and average profit on day t+7.

### Simulation for  t+1  Predictions

```python
# Initialize seed money
seed_money = 10000

# Initialize lists to store profits
profits_t1 = []

# Iterate through the dataset
for i in range(len(X_test_t1) - 1):  # t+1, so we use data up to the second to last day
    open_price = X_test_t1.iloc[i]['open']
    close_price = X_test_t1.iloc[i]['close']
    # prediction = y_pred_t1_best[i]
    prediction = log_reg_t1.predict(X_test_t1_scaled)[i]
  
    if prediction == 1:  # If the model predicts the stock will go up
        profit = (close_price - open_price) / open_price * seed_money
        profits_t1.append(profit)
    else:
        profits_t1.append(0)  # No action, no profit

# Calculate the average profit for t+1 predictions
average_profit_t1 = sum(profits_t1) / len(profits_t1)
average_profit_t1
```
-25.751654738086735

```python
# Initialize lists to store profits
profits_t7 = []

# Iterate through the dataset
for i in range(len(X_test_t7) - 7):  # t+7, so we use data up to the seventh to last day
    open_price = X_test_t7.iloc[i]['open']
    close_price = X_test_t7.iloc[i + 7]['close']  # t+7
    prediction = y_pred_t7_best[i]
  
    if prediction == 1:  # If the model predicts the stock will go up
        profit = (close_price - open_price) / open_price * seed_money
        profits_t7.append(profit)
    else:
        profits_t7.append(0)  # No action, no profit

# Calculate the average profit for t+7 predictions
average_profit_t7 = sum(profits_t7) / len(profits_t7)
average_profit_t7

```
6230.212552126924

## Market Simulation Visualization

```python
import matplotlib.pyplot as plt

# Plot profit distribution for t+1
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.hist(profits_t1, bins=30, edgecolor='k', alpha=0.7)
plt.title('Profit Distribution for t+1 Predictions')
plt.xlabel('Profit ($)')
plt.ylabel('Frequency')

# Plot profit distribution for t+7
plt.subplot(1, 2, 2)
plt.hist(profits_t7, bins=30, edgecolor='k', alpha=0.7)
plt.title('Profit Distribution for t+7 Predictions')
plt.xlabel('Profit ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```
![png](tweet_stock_files/tweet_stock_181_0.png)

```python
y_pred_t1_best
```
array([1, 1, 1, ..., 1, 1, 0])

```python
# Get the actual and predicted closing prices for t+7
actual_prices_t7 = y_.iloc[len(X_train_t7_scaled):]['close'].values[:-7]  # Actual closing prices
predicted_prices_t7 = data.iloc[len(X_train_t7_scaled):]['open'].values[:-7] * (1 + y_pred_t7_best)  # Predicted closing prices based on predictions

# Plot actual vs. predicted closing prices for t+7
plt.figure(figsize=(14, 7))
plt.plot(actual_prices_t7, label='Actual Prices')
plt.plot(predicted_prices_t7, label='Predicted Prices', linestyle='--')
plt.title('Actual vs. Predicted Closing Prices for t+7 Predictions')
plt.xlabel('Days')
plt.ylabel('Closing Price ($)')
plt.legend()
plt.show()
```
---

NameError                                 Traceback (most recent call last)

Cell In[407], line 2
1 # Get the actual and predicted closing prices for t+7
----> 2 actual_prices_t7 = y_.iloc[len(X_train_t7_scaled):]['close'].values[:-7]  # Actual closing prices
3 predicted_prices_t7 = data.iloc[len(X_train_t7_scaled):]['open'].values[:-7] * (1 + y_pred_t7_best)  # Predicted closing prices based on predictions
5 # Plot actual vs. predicted closing prices for t+7

NameError: name 'y_' is not defined
