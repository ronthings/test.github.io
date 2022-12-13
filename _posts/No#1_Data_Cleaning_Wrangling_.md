# Capstone Project - Predicting The Risk Of Customer Churn 

## Data Cleaning and Wrangling 
***

### Author - Oyeronke Ayansola

### Date - 12/12/2022

### <font color=red> Notebook #1</font>

***

## Table of Contents
### [Problem Statement](##Problemstatement)
### [Data Source And Overview](##DataSourceAndOverview)
### [Importing Libraries And Datasets](##ImportingLibrariesAndDatasets)
### [Explore Dataframes](##ExploreAndMergeDataframes)
   ### [1.Customer Data](##1_CustomerData)
   ### [2.Location Data](##2_LocationData)
   ### [3.Order Items Data](##3_OrderItemsData)
   ### [4.Payment Data](##4_PaymentData)
   ### [5.Review Data](##5_ReviewData)
   ### [6.Order Data](##6_OrderData)
   ### [7.ProductData](##7_ProductData)
   ### [8.Seller Data](##8_SellerData)
   ### [9.Product Name Data](##9_ProductNameData)
### [Merge Dataframes](##MergeDataframes)
### [Data Cleaning](##DataCleaning)
### [Conclusion](##Conclusion)


***
## Problem Statement 

**By exploring a given data and using machine learning algorithms we want to predict the customers are likely to churn with goal of improving customer services and overall business growth.**

***

## Data Source And Overview 

The data is a Brazilian ecommerce public dataset of orders made at **Olist Store.** The dataset contain orders information from late 2016 to 2018 with an average of 100,000 orders made at multiple marketplaces in Brazil.

Beacuse the data is a  real commercial data, it was anonymised, and references to the companies and partners in the review text was replaced with the names of Game of Thrones great houses.

The whole data is made up of nine different dataframes:
* Customer dataframe - This dataset contains customers information and location
* Order dataframe - The order dataset has information on all orders made by customers 
* Order Item dataframe - It contains information about items purchased 
* Location dataframe - The dataframe has information about Brazil geographic location by coordinates, Zip-codes of states and cities
* Sellers dataframe - Sellers location and information on sellers may be found here
* Products dataframe - This dataset contains information about all products
* Product name in English - This contains translation of products name listed in Portugese (Brazilian official language) to English language
* Payment dataframe - All information about payment methods and value may be found here
* Review dataframe - This contains reviews made by customer per item

The different dataframes were joined into a main dataframe as describe in the schema shown below.

***

![Screenshot%202022-11-30%20at%2021.05.42.png](attachment:Screenshot%202022-11-30%20at%2021.05.42.png)

Link to dataset on Kaggle:https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce


***
## Importing Libraries and Datasets <a name="ImportingLibrariesAndDatasets"></a>

The first step is to import the essential libraries that we will use in this notebook.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
```

Next step is to load the datasets from different comma separated files (csv) using pandas `read.csv`


```python
# loading our csv files 
customer_df = pd.read_csv('data/olist_customers_dataset.csv')
location_df = pd.read_csv('data/olist_geolocation_dataset.csv')
order_items_df = pd.read_csv('data/olist_order_items_dataset.csv')
payments_df = pd.read_csv('data/olist_order_payments_dataset.csv')
review_df = pd.read_csv('data/olist_order_reviews_dataset.csv')
orders_df = pd.read_csv ('data/olist_orders_dataset.csv')
products_df = pd.read_csv('data/olist_products_dataset.csv')
sellers_df = pd.read_csv('data/olist_sellers_dataset.csv')
prodct_name_eng_df = pd.read_csv ('data/product_category_name_translation.csv')
```

***
## Explore and Merge Data Frames

### 1. Customer data

We will explore all the dataframe individually and merge dataframe where neccesary. Let us start with customer dataframe `df_customer`. 


```python
# see the first few rows 
customer_df.head()
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
      <th>customer_id</th>
      <th>customer_unique_id</th>
      <th>customer_zip_code_prefix</th>
      <th>customer_city</th>
      <th>customer_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18955e83d337fd6b2def6b18a428ac77</td>
      <td>290c77bc529b7ac935b93aa66c333dc3</td>
      <td>9790</td>
      <td>sao bernardo do campo</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4e7b3e00288586ebd08712fdd0374a03</td>
      <td>060e732b5b29e8181a18229c7b0b2b5e</td>
      <td>1151</td>
      <td>sao paulo</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b2b6027bc5c5109e529d4dc6358b12c3</td>
      <td>259dac757896d24d7702b9acbbff3f3c</td>
      <td>8775</td>
      <td>mogi das cruzes</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4f2d8ab171c80ec8364f7c12e35b23ad</td>
      <td>345ecd01c38d18a9036ed96c73b8d066</td>
      <td>13056</td>
      <td>campinas</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the customer data types 
customer_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 99441 entries, 0 to 99440
    Data columns (total 5 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   customer_id               99441 non-null  object
     1   customer_unique_id        99441 non-null  object
     2   customer_zip_code_prefix  99441 non-null  int64 
     3   customer_city             99441 non-null  object
     4   customer_state            99441 non-null  object
    dtypes: int64(1), object(4)
    memory usage: 3.8+ MB



```python
# check the data shape
customer_df.shape
```




    (99441, 5)



We have 99,441 rows and 5 columns. There are two columns with customer Identities, `customer_id` and `customer_unique_id`. The `customer_unique_id` is unique to each customer while `customer_id` is generated for every order. It is important to check if the statement is true because in order to understand and predict churn, `customer_id` for individual order must not be duplicated.  


```python
# check for duplicated values in 'customer_id' and 'customer_unique_id'
print(f" Customer ID for each order : {customer_df['customer_id'].duplicated().value_counts()}")
print(f"Customer unique identifier for each customer :{customer_df['customer_unique_id'].duplicated().value_counts()}")
```

     Customer ID for each order : False    99441
    Name: customer_id, dtype: int64
    Customer unique identifier for each customer :False    96096
    True      3345
    Name: customer_unique_id, dtype: int64


There is no duplication of data in our customer unique identifier and customer ID for each order. `customer_id` has 99,441 rows, same lenght as our dataframe, that means we have customer order IDs generated for all order. `customer_unique_id` contains 96,096 rows, less than our dataframe lenght, this implies we have returning customers.


```python
# check the dataframe for null values
customer_df.isnull().sum()
```




    customer_id                 0
    customer_unique_id          0
    customer_zip_code_prefix    0
    customer_city               0
    customer_state              0
    dtype: int64



### 2. Location data


```python
# see the first few rows 
location_df.head()
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
      <th>geolocation_zip_code_prefix</th>
      <th>geolocation_lat</th>
      <th>geolocation_lng</th>
      <th>geolocation_city</th>
      <th>geolocation_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1037</td>
      <td>-23.545621</td>
      <td>-46.639292</td>
      <td>sao paulo</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1046</td>
      <td>-23.546081</td>
      <td>-46.644820</td>
      <td>sao paulo</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1046</td>
      <td>-23.546129</td>
      <td>-46.642951</td>
      <td>sao paulo</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1041</td>
      <td>-23.544392</td>
      <td>-46.639499</td>
      <td>sao paulo</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1035</td>
      <td>-23.541578</td>
      <td>-46.641607</td>
      <td>sao paulo</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>



The location data contains zip code, cities, states, longitude and latitude of the cities.


```python
# check the shape 
location_df.shape
```




    (1000163, 5)




```python
# check data types and info
location_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000163 entries, 0 to 1000162
    Data columns (total 5 columns):
     #   Column                       Non-Null Count    Dtype  
    ---  ------                       --------------    -----  
     0   geolocation_zip_code_prefix  1000163 non-null  int64  
     1   geolocation_lat              1000163 non-null  float64
     2   geolocation_lng              1000163 non-null  float64
     3   geolocation_city             1000163 non-null  object 
     4   geolocation_state            1000163 non-null  object 
    dtypes: float64(2), int64(1), object(2)
    memory usage: 38.2+ MB



```python
# check the dataframe for null values
location_df.isnull().sum()
```




    geolocation_zip_code_prefix    0
    geolocation_lat                0
    geolocation_lng                0
    geolocation_city               0
    geolocation_state              0
    dtype: int64



### 3. Order Item data


```python
# see the first few rows
order_items_df.head()
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
      <th>order_id</th>
      <th>order_item_id</th>
      <th>product_id</th>
      <th>seller_id</th>
      <th>shipping_limit_date</th>
      <th>price</th>
      <th>freight_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00010242fe8c5a6d1ba2dd792cb16214</td>
      <td>1</td>
      <td>4244733e06e7ecb4970a6e2683c13e61</td>
      <td>48436dade18ac8b2bce089ec2a041202</td>
      <td>19/09/2017 09:45</td>
      <td>58.90</td>
      <td>13.29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00018f77f2f0320c557190d7a144bdd3</td>
      <td>1</td>
      <td>e5f2d52b802189ee658865ca93d83a8f</td>
      <td>dd7ddc04e1b6c2c614352b383efe2d36</td>
      <td>03/05/2017 11:05</td>
      <td>239.90</td>
      <td>19.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000229ec398224ef6ca0657da4fc703e</td>
      <td>1</td>
      <td>c777355d18b72b67abbeef9df44fd0fd</td>
      <td>5b51032eddd242adc84c38acab88f23d</td>
      <td>18/01/2018 14:48</td>
      <td>199.00</td>
      <td>17.87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00024acbcdf0a6daa1e931b038114c75</td>
      <td>1</td>
      <td>7634da152a4610f1595efa32f14722fc</td>
      <td>9d7a1d34a5052409006425275ba1c2b4</td>
      <td>15/08/2018 10:10</td>
      <td>12.99</td>
      <td>12.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00042b26cf59d7ce69dfabb4e55b4fd9</td>
      <td>1</td>
      <td>ac6c3623068f30de03045865e4e10089</td>
      <td>df560393f3a51e74553ab94004ba5c87</td>
      <td>13/02/2017 13:57</td>
      <td>199.90</td>
      <td>18.14</td>
    </tr>
  </tbody>
</table>
</div>



The dataframe here contains IDs of seller, products, order items ids which is unique to each item, order ids, the product price, date requires for shipping and freight value.


```python
# check data types and info
order_items_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 112650 entries, 0 to 112649
    Data columns (total 7 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   order_id             112650 non-null  object 
     1   order_item_id        112650 non-null  int64  
     2   product_id           112650 non-null  object 
     3   seller_id            112650 non-null  object 
     4   shipping_limit_date  112650 non-null  object 
     5   price                112650 non-null  float64
     6   freight_value        112650 non-null  float64
    dtypes: float64(2), int64(1), object(4)
    memory usage: 6.0+ MB



```python
# check the shape 
order_items_df.shape
```




    (112650, 7)



In the order_items dataframe, we have 112,650 order rows and 7 columns for all items ordered. 


```python
# check for duplicated values in 'customer_id' and 'customer_unique_id'
print(f" ID per order : {order_items_df['order_id'].duplicated().value_counts()}")
print(f"Customer unique identifier for each customer :{order_items_df['order_item_id'].duplicated().value_counts()}")
```

     ID per order : False    98666
    True     13984
    Name: order_id, dtype: int64
    Customer unique identifier for each customer :True     112629
    False        21
    Name: order_item_id, dtype: int64


### 4. Payment data


```python
# see the first few rows
payments_df.head()
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
      <th>order_id</th>
      <th>payment_sequential</th>
      <th>payment_type</th>
      <th>payment_installments</th>
      <th>payment_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b81ef226f3fe1789b1e8b2acac839d17</td>
      <td>1</td>
      <td>credit_card</td>
      <td>8</td>
      <td>99.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a9810da82917af2d9aefd1278f1dcfa0</td>
      <td>1</td>
      <td>credit_card</td>
      <td>1</td>
      <td>24.39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25e8ea4e93396b6fa0d3dd708e76c1bd</td>
      <td>1</td>
      <td>credit_card</td>
      <td>1</td>
      <td>65.71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ba78997921bbcdc1373bb41e913ab953</td>
      <td>1</td>
      <td>credit_card</td>
      <td>8</td>
      <td>107.78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42fdf880ba16b47b59251dd489d4441a</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2</td>
      <td>128.45</td>
    </tr>
  </tbody>
</table>
</div>



The payment data has information about paymanet installments, payment value, how payment was made and the sequence.


```python
# check null values 
payments_df.isnull().sum()
```




    order_id                0
    payment_sequential      0
    payment_type            0
    payment_installments    0
    payment_value           0
    dtype: int64




```python
# see the types of payment methods
payments_df['payment_type'].unique()
```




    array(['credit_card', 'boleto', 'voucher', 'debit_card', 'not_defined'],
          dtype=object)




```python
# check the shape of the data
payments_df.shape
```




    (103886, 5)



### 5. Review data


```python
# see the first few rows
review_df.head()
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
      <th>review_id</th>
      <th>order_id</th>
      <th>review_score</th>
      <th>review_comment_title</th>
      <th>review_comment_message</th>
      <th>review_creation_date</th>
      <th>review_answer_timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7bc2406110b926393aa56f80a40eba40</td>
      <td>73fc7af87114b39712e6da79b0a377eb</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18/01/2018 00:00</td>
      <td>18/01/2018 21:46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80e641a11e56f04c1ad469d5645fdfde</td>
      <td>a548910a1c6147796b98fdf73dbeba33</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10/03/2018 00:00</td>
      <td>11/03/2018 03:05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>228ce5500dc1d8e020d8d1322874b6f0</td>
      <td>f9e4b658b201a9f2ecdecbb34bed034b</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17/02/2018 00:00</td>
      <td>18/02/2018 14:36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e64fb393e7b32834bb789ff8bb30750e</td>
      <td>658677c97b385a9be170737859d3511b</td>
      <td>5</td>
      <td>NaN</td>
      <td>Recebi bem antes do prazo estipulado.</td>
      <td>21/04/2017 00:00</td>
      <td>21/04/2017 22:02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f7c4243c7fe1938f181bec41a392bdeb</td>
      <td>8e6bfb81e283fa7e4f11123a3fb894f1</td>
      <td>5</td>
      <td>NaN</td>
      <td>Parabéns lojas lannister adorei comprar pela I...</td>
      <td>01/03/2018 00:00</td>
      <td>02/03/2018 10:26</td>
    </tr>
  </tbody>
</table>
</div>



Review dataframe has the review score, review messages with titles, the time review was created and answered. 


```python
# check the dataframe shape
review_df.shape
```




    (99224, 7)




```python
# check the null values in review data
review_df.isnull().sum()
```




    review_id                      0
    order_id                       0
    review_score                   0
    review_comment_title       87656
    review_comment_message     58247
    review_creation_date           0
    review_answer_timestamp        0
    dtype: int64



The `review_comment_title` and `review_comment_message` has a lot of null values present. Let us check the pertange of the null values, so we will have an insight to how to deal with null values.


```python
# check the percentage of null values in review dataframe
review_df.isnull().sum()/review_df.shape[0]*100
```




    review_id                   0.000000
    order_id                    0.000000
    review_score                0.000000
    review_comment_title       88.341530
    review_comment_message     58.702532
    review_creation_date        0.000000
    review_answer_timestamp     0.000000
    dtype: float64



For the columns with null values, `review_comment_message` has 58.7 % null values and `review_comment_title` with 88.34%. Because they both have more than 50% of null values, we will drop the two columns.


```python
# drop `review_comment_title` and `review_comment_message` 
review_df.dropna(axis = 'columns', inplace = True)
```


```python
# recheck if the columns were dropped
review_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 99224 entries, 0 to 99223
    Data columns (total 5 columns):
     #   Column                   Non-Null Count  Dtype 
    ---  ------                   --------------  ----- 
     0   review_id                99224 non-null  object
     1   order_id                 99224 non-null  object
     2   review_score             99224 non-null  int64 
     3   review_creation_date     99224 non-null  object
     4   review_answer_timestamp  99224 non-null  object
    dtypes: int64(1), object(4)
    memory usage: 3.8+ MB


### 6. Order data


```python
# check the first few rows
orders_df.head()
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
      <th>order_id</th>
      <th>customer_id</th>
      <th>order_status</th>
      <th>order_purchase_timestamp</th>
      <th>order_approved_at</th>
      <th>order_delivered_carrier_date</th>
      <th>order_delivered_customer_date</th>
      <th>order_estimated_delivery_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>e481f51cbdc54678b7cc49136f2d6af7</td>
      <td>9ef432eb6251297304e76186b10a928d</td>
      <td>delivered</td>
      <td>2017-10-02 10:56:33</td>
      <td>2017-10-02 11:07:15</td>
      <td>2017-10-04 19:55:00</td>
      <td>2017-10-10 21:25:13</td>
      <td>2017-10-18 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53cdb2fc8bc7dce0b6741e2150273451</td>
      <td>b0830fb4747a6c6d20dea0b8c802d7ef</td>
      <td>delivered</td>
      <td>2018-07-24 20:41:37</td>
      <td>2018-07-26 03:24:27</td>
      <td>2018-07-26 14:31:00</td>
      <td>2018-08-07 15:27:45</td>
      <td>2018-08-13 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47770eb9100c2d0c44946d9cf07ec65d</td>
      <td>41ce2a54c0b03bf3443c3d931a367089</td>
      <td>delivered</td>
      <td>2018-08-08 08:38:49</td>
      <td>2018-08-08 08:55:23</td>
      <td>2018-08-08 13:50:00</td>
      <td>2018-08-17 18:06:29</td>
      <td>2018-09-04 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>949d5b44dbf5de918fe9c16f97b45f8a</td>
      <td>f88197465ea7920adcdbec7375364d82</td>
      <td>delivered</td>
      <td>2017-11-18 19:28:06</td>
      <td>2017-11-18 19:45:59</td>
      <td>2017-11-22 13:39:59</td>
      <td>2017-12-02 00:28:42</td>
      <td>2017-12-15 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ad21c59c0840e6cb83a9ceb5573f8159</td>
      <td>8ab97904e6daea8866dbdbc4fb7aad2c</td>
      <td>delivered</td>
      <td>2018-02-13 21:18:39</td>
      <td>2018-02-13 22:20:29</td>
      <td>2018-02-14 19:46:34</td>
      <td>2018-02-16 18:17:02</td>
      <td>2018-02-26 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>



Order dataframe contains all information about orders: approved date and time, the status, when the order was purchased, estimated delicery date, date order was delivered, carrir delivered date.


```python
# see the shape 
orders_df.shape
```




    (99441, 8)




```python
# check for null values
orders_df.isnull().sum()
```




    order_id                            0
    customer_id                         0
    order_status                        0
    order_purchase_timestamp            0
    order_approved_at                 160
    order_delivered_carrier_date     1783
    order_delivered_customer_date    2965
    order_estimated_delivery_date       0
    dtype: int64



There are null values present in order_approved_at, order_delivered_carrier_date and order_delivered_customer_date. Let us check the percentages of missing values in the order dataframe.


```python
# check the percentage of null values in order dataframe
orders_df.isnull().sum()/orders_df.shape[0]*100
```




    order_id                         0.000000
    customer_id                      0.000000
    order_status                     0.000000
    order_purchase_timestamp         0.000000
    order_approved_at                0.160899
    order_delivered_carrier_date     1.793023
    order_delivered_customer_date    2.981668
    order_estimated_delivery_date    0.000000
    dtype: float64



From the percentages, `order_approved_at` has **0.16%** missing values, `order_delivered_carrier_date` has **1.79%** and `order_delivered_customer_date` with **2.98%**. These are negligible percentages therefore if we fill in the missing values using a `forward fill` or `backward fill` the interpretation of our result will not be altered. Therefore, we will `forward fill` the missing values.


```python
# forward fill missing values

orders_df["order_approved_at"] = orders_df["order_approved_at"].ffill()
orders_df["order_delivered_customer_date"]= orders_df["order_delivered_customer_date"].ffill()
orders_df["order_delivered_carrier_date"]= orders_df["order_delivered_carrier_date"].ffill()
```


```python
# recheck for null values
orders_df.isnull().sum()
```




    order_id                         0
    customer_id                      0
    order_status                     0
    order_purchase_timestamp         0
    order_approved_at                0
    order_delivered_carrier_date     0
    order_delivered_customer_date    0
    order_estimated_delivery_date    0
    dtype: int64



We now have zero `NaNs` values in the order dataframe.

### 7. Products data


```python
# check the product data
products_df.head()
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
      <th>product_id</th>
      <th>product_category_name</th>
      <th>product_name_lenght</th>
      <th>product_description_lenght</th>
      <th>product_photos_qty</th>
      <th>product_weight_g</th>
      <th>product_length_cm</th>
      <th>product_height_cm</th>
      <th>product_width_cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1e9e8ef04dbcff4541ed26657ea517e5</td>
      <td>perfumaria</td>
      <td>40.0</td>
      <td>287.0</td>
      <td>1.0</td>
      <td>225.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3aa071139cb16b67ca9e5dea641aaa2f</td>
      <td>artes</td>
      <td>44.0</td>
      <td>276.0</td>
      <td>1.0</td>
      <td>1000.0</td>
      <td>30.0</td>
      <td>18.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>96bd76ec8810374ed1b65e291975717f</td>
      <td>esporte_lazer</td>
      <td>46.0</td>
      <td>250.0</td>
      <td>1.0</td>
      <td>154.0</td>
      <td>18.0</td>
      <td>9.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cef67bcfe19066a932b7673e239eb23d</td>
      <td>bebes</td>
      <td>27.0</td>
      <td>261.0</td>
      <td>1.0</td>
      <td>371.0</td>
      <td>26.0</td>
      <td>4.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9dc1a7de274444849c219cff195d0b71</td>
      <td>utilidades_domesticas</td>
      <td>37.0</td>
      <td>402.0</td>
      <td>4.0</td>
      <td>625.0</td>
      <td>20.0</td>
      <td>17.0</td>
      <td>13.0</td>
    </tr>
  </tbody>
</table>
</div>



The product dataframe contains information about product - the weight, lenght, height, width, product name lenght,the product categoryand photos quantity. 


```python
# see the shape of the data
products_df.shape
```




    (32951, 9)




```python
# check the null values in the data
products_df.isnull().sum()
```




    product_id                      0
    product_category_name         610
    product_name_lenght           610
    product_description_lenght    610
    product_photos_qty            610
    product_weight_g                2
    product_length_cm               2
    product_height_cm               2
    product_width_cm                2
    dtype: int64



We have some null values in all the columns except `product_id`. Let us check the percentage, it will give us an insight to how to deal with the nulls.


```python
# check the percentage of null values in order dataframe
products_df.isnull().sum()/products_df.shape[0]*100
```




    product_id                    0.000000
    product_category_name         1.851234
    product_name_lenght           1.851234
    product_description_lenght    1.851234
    product_photos_qty            1.851234
    product_weight_g              0.006070
    product_length_cm             0.006070
    product_height_cm             0.006070
    product_width_cm              0.006070
    dtype: float64



The percentages are negligible. Let us check if the null values columns.


```python
products_df[products_df[['product_category_name', 'product_name_lenght', 
                         'product_description_lenght', 'product_photos_qty']].isnull().any(axis=1)]
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
      <th>product_id</th>
      <th>product_category_name</th>
      <th>product_name_lenght</th>
      <th>product_description_lenght</th>
      <th>product_photos_qty</th>
      <th>product_weight_g</th>
      <th>product_length_cm</th>
      <th>product_height_cm</th>
      <th>product_width_cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105</th>
      <td>a41e356c76fab66334f36de622ecbd3a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>650.0</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>128</th>
      <td>d8dee61c2034d6d075997acef1870e9b</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>300.0</td>
      <td>16.0</td>
      <td>7.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>145</th>
      <td>56139431d72cd51f19eb9f7dae4d1617</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200.0</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>154</th>
      <td>46b48281eb6d663ced748f324108c733</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18500.0</td>
      <td>41.0</td>
      <td>30.0</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>197</th>
      <td>5fb61f482620cb672f5e586bb132eae9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>300.0</td>
      <td>35.0</td>
      <td>7.0</td>
      <td>12.0</td>
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
    </tr>
    <tr>
      <th>32515</th>
      <td>b0a0c5dd78e644373b199380612c350a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1800.0</td>
      <td>30.0</td>
      <td>20.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>32589</th>
      <td>10dbe0fbaa2c505123c17fdc34a63c56</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>800.0</td>
      <td>30.0</td>
      <td>10.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>32616</th>
      <td>bd2ada37b58ae94cc838b9c0569fecd8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200.0</td>
      <td>21.0</td>
      <td>8.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>32772</th>
      <td>fa51e914046aab32764c41356b9d4ea4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1300.0</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>32852</th>
      <td>c4ceee876c82b8328e9c293fa0e1989b</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>700.0</td>
      <td>28.0</td>
      <td>3.0</td>
      <td>43.0</td>
    </tr>
  </tbody>
</table>
<p>610 rows × 9 columns</p>
</div>



The values are similar across the product_category_name, product_name_lenght, product_description_lenght, product_photos_qty. We will drop the nulls values.


```python
# drop the missing values
products_df.dropna(inplace = True)
```


```python
# sanity check
products_df.isnull().sum()
```




    product_id                    0
    product_category_name         0
    product_name_lenght           0
    product_description_lenght    0
    product_photos_qty            0
    product_weight_g              0
    product_length_cm             0
    product_height_cm             0
    product_width_cm              0
    dtype: int64



There is no null values anymore in the product data

### 8. Sellers data


```python
# check the sellers data
sellers_df.head()
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
      <th>seller_id</th>
      <th>seller_zip_code_prefix</th>
      <th>seller_city</th>
      <th>seller_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d1b65fc7debc3361ea86b5f14c68d2e2</td>
      <td>13844</td>
      <td>mogi guacu</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ce3ad9de960102d0677a81f5d0bb7b2d</td>
      <td>20031</td>
      <td>rio de janeiro</td>
      <td>RJ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c0f3eea2e14555b6faeea3dd58c1b1c3</td>
      <td>4195</td>
      <td>sao paulo</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51a04a8a6bdcb23deccc82b0b80742cf</td>
      <td>12914</td>
      <td>braganca paulista</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>



Sellers data has information on sellers location: the cities, states and zip codes.


```python
# see the shape of the data
sellers_df.shape
```




    (3095, 4)




```python
# check null values
sellers_df.isnull().sum()
```




    seller_id                 0
    seller_zip_code_prefix    0
    seller_city               0
    seller_state              0
    dtype: int64



### 9. Product name data


```python
# check the data
prodct_name_eng_df.head()
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
      <th>product_category_name</th>
      <th>product_category_name_english</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>beleza_saude</td>
      <td>health_beauty</td>
    </tr>
    <tr>
      <th>1</th>
      <td>informatica_acessorios</td>
      <td>computers_accessories</td>
    </tr>
    <tr>
      <th>2</th>
      <td>automotivo</td>
      <td>auto</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cama_mesa_banho</td>
      <td>bed_bath_table</td>
    </tr>
    <tr>
      <th>4</th>
      <td>moveis_decoracao</td>
      <td>furniture_decor</td>
    </tr>
  </tbody>
</table>
</div>



This is a two column dataframe with product category name in portuguese and english.


```python
# check the shape
prodct_name_eng_df.shape
```




    (71, 2)




```python
# check for null values
prodct_name_eng_df.isnull().sum()
```




    product_category_name            0
    product_category_name_english    0
    dtype: int64



### Merge Data Frame

Next, we will join all the differrent dataframes together by first joining location to customer dataframe in order to add the customer coordinates (latitude and longitude)


```python
# join location and customer dataframe
customer_location_df = pd.merge(customer_df, location_df , left_on='customer_zip_code_prefix',
                               right_on='geolocation_zip_code_prefix' )
```


```python
# check the new customer location
customer_location_df.head()
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
      <th>customer_id</th>
      <th>customer_unique_id</th>
      <th>customer_zip_code_prefix</th>
      <th>customer_city</th>
      <th>customer_state</th>
      <th>geolocation_zip_code_prefix</th>
      <th>geolocation_lat</th>
      <th>geolocation_lng</th>
      <th>geolocation_city</th>
      <th>geolocation_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>14409</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>14409</td>
      <td>-20.497396</td>
      <td>-47.399241</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>14409</td>
      <td>-20.510459</td>
      <td>-47.399553</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>14409</td>
      <td>-20.480940</td>
      <td>-47.394161</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>14409</td>
      <td>-20.515413</td>
      <td>-47.398194</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the sahpe of customer location
customer_location_df.shape
```




    (15083455, 10)



Netx, we will drop `geolocation_zip_code_prefix` and rename `geolocation_lat` and `geolocation_lng` to reflect customers location


```python
# drop 'geolocation_zip_code_prefix' column
customer_location_df.drop('geolocation_zip_code_prefix', axis=1, inplace=True)
```


```python
# check the customer location
customer_location_df.head()
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
      <th>customer_id</th>
      <th>customer_unique_id</th>
      <th>customer_zip_code_prefix</th>
      <th>customer_city</th>
      <th>customer_state</th>
      <th>geolocation_lat</th>
      <th>geolocation_lng</th>
      <th>geolocation_city</th>
      <th>geolocation_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.497396</td>
      <td>-47.399241</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.510459</td>
      <td>-47.399553</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.480940</td>
      <td>-47.394161</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.515413</td>
      <td>-47.398194</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rename 'geolocation_lng' and 'geolocation_lat'
customer_location_df = customer_location_df.rename(columns = {'geolocation_lat' : 'customer_latitude', 
                                                              'geolocation_lng': 'customer_longitude'})
```


```python
# recheck the dataframe
customer_location_df.head()
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
      <th>customer_id</th>
      <th>customer_unique_id</th>
      <th>customer_zip_code_prefix</th>
      <th>customer_city</th>
      <th>customer_state</th>
      <th>customer_latitude</th>
      <th>customer_longitude</th>
      <th>geolocation_city</th>
      <th>geolocation_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.497396</td>
      <td>-47.399241</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.510459</td>
      <td>-47.399553</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.480940</td>
      <td>-47.394161</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.515413</td>
      <td>-47.398194</td>
      <td>franca</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop 'geolocation_city' and 'geolocation_state' to avoid repetition
customer_location_df.drop(['geolocation_city', 'geolocation_state'], axis=1, inplace=True)
```


```python
# recheck the dataframe
customer_location_df.head()
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
      <th>customer_id</th>
      <th>customer_unique_id</th>
      <th>customer_zip_code_prefix</th>
      <th>customer_city</th>
      <th>customer_state</th>
      <th>customer_latitude</th>
      <th>customer_longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.497396</td>
      <td>-47.399241</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.510459</td>
      <td>-47.399553</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.480940</td>
      <td>-47.394161</td>
    </tr>
    <tr>
      <th>4</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.515413</td>
      <td>-47.398194</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the shape of the dataframe
customer_location_df.shape
```




    (15083455, 7)



The new customer location dataframe has 15,083,455 rows and 7 columns. That means we have some duplication when joining the dataframes.


```python
# check for duplicates
customer_location_df['customer_id'].duplicated().sum()
```




    14984292




```python
# remove all the duplicates
customer_location_df = customer_location_df.drop_duplicates(subset=["customer_id"], keep = 'first')
```


```python
# recheck for duplicates
customer_location_df['customer_id'].duplicated().sum()
```




    0




```python
# recheck the shape
customer_location_df.shape
```




    (99163, 7)



Let us do same for Sellers location


```python
# merge location and sellers dataframe 
sellers_location_df = pd.merge(sellers_df, location_df , left_on='seller_zip_code_prefix', 
                          right_on='geolocation_zip_code_prefix')
```


```python
# check the new sellers location dataframe
sellers_location_df.head()
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
      <th>seller_id</th>
      <th>seller_zip_code_prefix</th>
      <th>seller_city</th>
      <th>seller_state</th>
      <th>geolocation_zip_code_prefix</th>
      <th>geolocation_lat</th>
      <th>geolocation_lng</th>
      <th>geolocation_city</th>
      <th>geolocation_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>13023</td>
      <td>-22.898536</td>
      <td>-47.063125</td>
      <td>campinas</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>13023</td>
      <td>-22.895499</td>
      <td>-47.061944</td>
      <td>campinas</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>13023</td>
      <td>-22.891740</td>
      <td>-47.060820</td>
      <td>campinas</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>13023</td>
      <td>-22.895762</td>
      <td>-47.066144</td>
      <td>campinas</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>13023</td>
      <td>-22.896154</td>
      <td>-47.062431</td>
      <td>campinas</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the dataframe shape
sellers_location_df.shape
```




    (435087, 9)




```python
# drop `geolocation_zip_code_prefix` , 
sellers_location_df.drop(['geolocation_zip_code_prefix', 'geolocation_city', 'geolocation_state'], 
                         axis=1, inplace=True)
```


```python
# check the dataframe
sellers_location_df.head()
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
      <th>seller_id</th>
      <th>seller_zip_code_prefix</th>
      <th>seller_city</th>
      <th>seller_state</th>
      <th>geolocation_lat</th>
      <th>geolocation_lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.898536</td>
      <td>-47.063125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.895499</td>
      <td>-47.061944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.891740</td>
      <td>-47.060820</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.895762</td>
      <td>-47.066144</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.896154</td>
      <td>-47.062431</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rename `geolocation_lat` and `geolocation_lng` to reflect customers location
sellers_location_df = sellers_location_df.rename(columns = {'geolocation_lat' : 'seller_latitude', 
                                                            'geolocation_lng': 'seller_longitude'})
```


```python
# check the dataframe
sellers_location_df.head()
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
      <th>seller_id</th>
      <th>seller_zip_code_prefix</th>
      <th>seller_city</th>
      <th>seller_state</th>
      <th>seller_latitude</th>
      <th>seller_longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.898536</td>
      <td>-47.063125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.895499</td>
      <td>-47.061944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.891740</td>
      <td>-47.060820</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.895762</td>
      <td>-47.066144</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.896154</td>
      <td>-47.062431</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove duplicates rows
sellers_location_df= sellers_location_df.drop_duplicates(subset=["seller_id"], keep = 'first')
```


```python
# recheck the dataframe
sellers_location_df.shape
```




    (3088, 6)




```python
sellers_location_df['seller_id'].duplicated().sum()
```




    0




```python
# recheck the dataframe
sellers_location_df.head()
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
      <th>seller_id</th>
      <th>seller_zip_code_prefix</th>
      <th>seller_city</th>
      <th>seller_state</th>
      <th>seller_latitude</th>
      <th>seller_longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3442f8959a84dea7ee197c632cb2df15</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.898536</td>
      <td>-47.063125</td>
    </tr>
    <tr>
      <th>80</th>
      <td>e0eabded302882513ced4ea3eb0c7059</td>
      <td>13023</td>
      <td>campinas</td>
      <td>SP</td>
      <td>-22.898536</td>
      <td>-47.063125</td>
    </tr>
    <tr>
      <th>160</th>
      <td>d1b65fc7debc3361ea86b5f14c68d2e2</td>
      <td>13844</td>
      <td>mogi guacu</td>
      <td>SP</td>
      <td>-22.382941</td>
      <td>-46.946641</td>
    </tr>
    <tr>
      <th>263</th>
      <td>ce3ad9de960102d0677a81f5d0bb7b2d</td>
      <td>20031</td>
      <td>rio de janeiro</td>
      <td>RJ</td>
      <td>-22.910641</td>
      <td>-43.176510</td>
    </tr>
    <tr>
      <th>650</th>
      <td>1d2732ef8321502ee8488e8bed1ab8cd</td>
      <td>20031</td>
      <td>rio de janeiro</td>
      <td>RJ</td>
      <td>-22.910641</td>
      <td>-43.176510</td>
    </tr>
  </tbody>
</table>
</div>



We will merge all the dataframe into one dataframe


```python
# join all the dataframe 
df = customer_location_df.merge(orders_df, on = 'customer_id', how = 'left')\
    .merge(payments_df, on = 'order_id', how = 'left') \
    .merge(order_items_df, on = 'order_id', how = 'left')\
    .merge(review_df, on = 'order_id', how = 'left')\
    .merge(products_df, on = 'product_id', how = 'left')\
    .merge(prodct_name_eng_df, on = 'product_category_name', how = 'left')\
    .merge(sellers_location_df, on = 'seller_id', how = 'left')
```


```python
# check the new dataframe
df.head()
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
      <th>customer_id</th>
      <th>customer_unique_id</th>
      <th>customer_zip_code_prefix</th>
      <th>customer_city</th>
      <th>customer_state</th>
      <th>customer_latitude</th>
      <th>customer_longitude</th>
      <th>order_id</th>
      <th>order_status</th>
      <th>order_purchase_timestamp</th>
      <th>...</th>
      <th>product_weight_g</th>
      <th>product_length_cm</th>
      <th>product_height_cm</th>
      <th>product_width_cm</th>
      <th>product_category_name_english</th>
      <th>seller_zip_code_prefix</th>
      <th>seller_city</th>
      <th>seller_state</th>
      <th>seller_latitude</th>
      <th>seller_longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>00e7ee1b050b8499577073aeb2a297a1</td>
      <td>delivered</td>
      <td>2017-05-16 15:05:35</td>
      <td>...</td>
      <td>8683.0</td>
      <td>54.0</td>
      <td>64.0</td>
      <td>31.0</td>
      <td>office_furniture</td>
      <td>8577.0</td>
      <td>itaquaquecetuba</td>
      <td>SP</td>
      <td>-23.482623</td>
      <td>-46.374490</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5dca924cc99eea2dc5ba40d11ec5dd0f</td>
      <td>2761fee7f378f0a8d7682d8a3fa07ab1</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>95261f608a64bbbe760a212b4d48a4ee</td>
      <td>delivered</td>
      <td>2018-06-15 20:07:13</td>
      <td>...</td>
      <td>1383.0</td>
      <td>50.0</td>
      <td>10.0</td>
      <td>40.0</td>
      <td>bed_bath_table</td>
      <td>14940.0</td>
      <td>ibitinga</td>
      <td>SP</td>
      <td>-21.766477</td>
      <td>-48.831547</td>
    </tr>
    <tr>
      <th>2</th>
      <td>661897d4968f1b59bfff74c7eb2eb4fc</td>
      <td>d06a495406b79cb8203ea21cc0942f8c</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>9444fa2ab50a3f5af63b48be297eda24</td>
      <td>delivered</td>
      <td>2017-09-09 15:40:00</td>
      <td>...</td>
      <td>1200.0</td>
      <td>47.0</td>
      <td>7.0</td>
      <td>26.0</td>
      <td>toys</td>
      <td>89204.0</td>
      <td>joinville</td>
      <td>SC</td>
      <td>-26.283149</td>
      <td>-48.851285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>702b62324327ccba20f1be3465426437</td>
      <td>8b3d988f330c1d1c3332ccd440c147b7</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>dceb8e88274c6f42a88a76ed979eb817</td>
      <td>delivered</td>
      <td>2018-03-26 12:04:55</td>
      <td>...</td>
      <td>567.0</td>
      <td>19.0</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>auto</td>
      <td>4243.0</td>
      <td>sao paulo</td>
      <td>SP</td>
      <td>-23.626269</td>
      <td>-46.586534</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bdf997bae7ca819b0415f5174d6b4302</td>
      <td>866755e25db620f8d7e81b351a15bb2f</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>6ee1cea1b2edcc713f83ebfbccbc57f9</td>
      <td>delivered</td>
      <td>2018-07-29 20:39:20</td>
      <td>...</td>
      <td>300.0</td>
      <td>20.0</td>
      <td>7.0</td>
      <td>15.0</td>
      <td>telephony</td>
      <td>1212.0</td>
      <td>sao paulo</td>
      <td>SP</td>
      <td>-23.537511</td>
      <td>-46.637057</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>




```python
# check the shape
df.shape
```




    (118821, 42)




```python
# see the dataframe datatypes and information
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 118821 entries, 0 to 118820
    Data columns (total 42 columns):
     #   Column                         Non-Null Count   Dtype  
    ---  ------                         --------------   -----  
     0   customer_id                    118821 non-null  object 
     1   customer_unique_id             118821 non-null  object 
     2   customer_zip_code_prefix       118821 non-null  int64  
     3   customer_city                  118821 non-null  object 
     4   customer_state                 118821 non-null  object 
     5   customer_latitude              118821 non-null  float64
     6   customer_longitude             118821 non-null  float64
     7   order_id                       118821 non-null  object 
     8   order_status                   118821 non-null  object 
     9   order_purchase_timestamp       118821 non-null  object 
     10  order_approved_at              118821 non-null  object 
     11  order_delivered_carrier_date   118821 non-null  object 
     12  order_delivered_customer_date  118821 non-null  object 
     13  order_estimated_delivery_date  118821 non-null  object 
     14  payment_sequential             118818 non-null  float64
     15  payment_type                   118818 non-null  object 
     16  payment_installments           118818 non-null  float64
     17  payment_value                  118818 non-null  float64
     18  order_item_id                  117993 non-null  float64
     19  product_id                     117993 non-null  object 
     20  seller_id                      117993 non-null  object 
     21  shipping_limit_date            117993 non-null  object 
     22  price                          117993 non-null  float64
     23  freight_value                  117993 non-null  float64
     24  review_id                      117826 non-null  object 
     25  review_score                   117826 non-null  float64
     26  review_creation_date           117826 non-null  object 
     27  review_answer_timestamp        117826 non-null  object 
     28  product_category_name          116289 non-null  object 
     29  product_name_lenght            116289 non-null  float64
     30  product_description_lenght     116289 non-null  float64
     31  product_photos_qty             116289 non-null  float64
     32  product_weight_g               116289 non-null  float64
     33  product_length_cm              116289 non-null  float64
     34  product_height_cm              116289 non-null  float64
     35  product_width_cm               116289 non-null  float64
     36  product_category_name_english  116264 non-null  object 
     37  seller_zip_code_prefix         117729 non-null  float64
     38  seller_city                    117729 non-null  object 
     39  seller_state                   117729 non-null  object 
     40  seller_latitude                117729 non-null  float64
     41  seller_longitude               117729 non-null  float64
    dtypes: float64(19), int64(1), object(22)
    memory usage: 39.0+ MB


Let us check for null values in our combined dataframe.


```python
# check for null values
df.isnull().sum()
```




    customer_id                         0
    customer_unique_id                  0
    customer_zip_code_prefix            0
    customer_city                       0
    customer_state                      0
    customer_latitude                   0
    customer_longitude                  0
    order_id                            0
    order_status                        0
    order_purchase_timestamp            0
    order_approved_at                   0
    order_delivered_carrier_date        0
    order_delivered_customer_date       0
    order_estimated_delivery_date       0
    payment_sequential                  3
    payment_type                        3
    payment_installments                3
    payment_value                       3
    order_item_id                     828
    product_id                        828
    seller_id                         828
    shipping_limit_date               828
    price                             828
    freight_value                     828
    review_id                         995
    review_score                      995
    review_creation_date              995
    review_answer_timestamp           995
    product_category_name            2532
    product_name_lenght              2532
    product_description_lenght       2532
    product_photos_qty               2532
    product_weight_g                 2532
    product_length_cm                2532
    product_height_cm                2532
    product_width_cm                 2532
    product_category_name_english    2557
    seller_zip_code_prefix           1092
    seller_city                      1092
    seller_state                     1092
    seller_latitude                  1092
    seller_longitude                 1092
    dtype: int64




```python
# check the null values percentage 
df.isnull().sum()/df.shape[0]*100
```




    customer_id                      0.000000
    customer_unique_id               0.000000
    customer_zip_code_prefix         0.000000
    customer_city                    0.000000
    customer_state                   0.000000
    customer_latitude                0.000000
    customer_longitude               0.000000
    order_id                         0.000000
    order_status                     0.000000
    order_purchase_timestamp         0.000000
    order_approved_at                0.000000
    order_delivered_carrier_date     0.000000
    order_delivered_customer_date    0.000000
    order_estimated_delivery_date    0.000000
    payment_sequential               0.002525
    payment_type                     0.002525
    payment_installments             0.002525
    payment_value                    0.002525
    order_item_id                    0.696847
    product_id                       0.696847
    seller_id                        0.696847
    shipping_limit_date              0.696847
    price                            0.696847
    freight_value                    0.696847
    review_id                        0.837394
    review_score                     0.837394
    review_creation_date             0.837394
    review_answer_timestamp          0.837394
    product_category_name            2.130936
    product_name_lenght              2.130936
    product_description_lenght       2.130936
    product_photos_qty               2.130936
    product_weight_g                 2.130936
    product_length_cm                2.130936
    product_height_cm                2.130936
    product_width_cm                 2.130936
    product_category_name_english    2.151977
    seller_zip_code_prefix           0.919029
    seller_city                      0.919029
    seller_state                     0.919029
    seller_latitude                  0.919029
    seller_longitude                 0.919029
    dtype: float64



There are null values present in our dataframe, ranging from 0.26% to 2.84%, dropping these null values will not have material inpact on our data and interpretation. Therefore we will drop all null values in our data.


```python
# drop 'review_comment_message' and 'review_comment_title'
df.dropna(inplace = True)
```


```python
# recheck the dataframe shape
df.shape
```




    (115036, 42)




```python
# recheck the null values 
df.isnull().sum()
```




    customer_id                      0
    customer_unique_id               0
    customer_zip_code_prefix         0
    customer_city                    0
    customer_state                   0
    customer_latitude                0
    customer_longitude               0
    order_id                         0
    order_status                     0
    order_purchase_timestamp         0
    order_approved_at                0
    order_delivered_carrier_date     0
    order_delivered_customer_date    0
    order_estimated_delivery_date    0
    payment_sequential               0
    payment_type                     0
    payment_installments             0
    payment_value                    0
    order_item_id                    0
    product_id                       0
    seller_id                        0
    shipping_limit_date              0
    price                            0
    freight_value                    0
    review_id                        0
    review_score                     0
    review_creation_date             0
    review_answer_timestamp          0
    product_category_name            0
    product_name_lenght              0
    product_description_lenght       0
    product_photos_qty               0
    product_weight_g                 0
    product_length_cm                0
    product_height_cm                0
    product_width_cm                 0
    product_category_name_english    0
    seller_zip_code_prefix           0
    seller_city                      0
    seller_state                     0
    seller_latitude                  0
    seller_longitude                 0
    dtype: int64



We now have a dataframe with zero null value.


```python
# rename product_name_lenght and product_description_lenght
df.rename(columns = {'product_description_lenght': 'product_description_length', 
          'product_name_lenght': 'product_name_length'}, inplace = True)
```


```python
#check the datatypes of our new non-null value dataframe
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 115036 entries, 0 to 118820
    Data columns (total 42 columns):
     #   Column                         Non-Null Count   Dtype  
    ---  ------                         --------------   -----  
     0   customer_id                    115036 non-null  object 
     1   customer_unique_id             115036 non-null  object 
     2   customer_zip_code_prefix       115036 non-null  int64  
     3   customer_city                  115036 non-null  object 
     4   customer_state                 115036 non-null  object 
     5   customer_latitude              115036 non-null  float64
     6   customer_longitude             115036 non-null  float64
     7   order_id                       115036 non-null  object 
     8   order_status                   115036 non-null  object 
     9   order_purchase_timestamp       115036 non-null  object 
     10  order_approved_at              115036 non-null  object 
     11  order_delivered_carrier_date   115036 non-null  object 
     12  order_delivered_customer_date  115036 non-null  object 
     13  order_estimated_delivery_date  115036 non-null  object 
     14  payment_sequential             115036 non-null  float64
     15  payment_type                   115036 non-null  object 
     16  payment_installments           115036 non-null  float64
     17  payment_value                  115036 non-null  float64
     18  order_item_id                  115036 non-null  float64
     19  product_id                     115036 non-null  object 
     20  seller_id                      115036 non-null  object 
     21  shipping_limit_date            115036 non-null  object 
     22  price                          115036 non-null  float64
     23  freight_value                  115036 non-null  float64
     24  review_id                      115036 non-null  object 
     25  review_score                   115036 non-null  float64
     26  review_creation_date           115036 non-null  object 
     27  review_answer_timestamp        115036 non-null  object 
     28  product_category_name          115036 non-null  object 
     29  product_name_length            115036 non-null  float64
     30  product_description_length     115036 non-null  float64
     31  product_photos_qty             115036 non-null  float64
     32  product_weight_g               115036 non-null  float64
     33  product_length_cm              115036 non-null  float64
     34  product_height_cm              115036 non-null  float64
     35  product_width_cm               115036 non-null  float64
     36  product_category_name_english  115036 non-null  object 
     37  seller_zip_code_prefix         115036 non-null  float64
     38  seller_city                    115036 non-null  object 
     39  seller_state                   115036 non-null  object 
     40  seller_latitude                115036 non-null  float64
     41  seller_longitude               115036 non-null  float64
    dtypes: float64(19), int64(1), object(22)
    memory usage: 37.7+ MB


All our date and time column are object, let us convert to datetime data type


```python
# import datetime
from datetime import datetime

# assign all columns with date and time to a dataframe
date_time_columns =['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
                           'order_estimated_delivery_date','shipping_limit_date', 'review_creation_date', 'review_answer_timestamp']
```


```python
# convert all date columns to datetime data type
df[date_time_columns] = df[date_time_columns].apply(pd.to_datetime)
```


```python
# recheck the datatype
df.dtypes
```




    customer_id                              object
    customer_unique_id                       object
    customer_zip_code_prefix                  int64
    customer_city                            object
    customer_state                           object
    customer_latitude                       float64
    customer_longitude                      float64
    order_id                                 object
    order_status                             object
    order_purchase_timestamp         datetime64[ns]
    order_approved_at                datetime64[ns]
    order_delivered_carrier_date     datetime64[ns]
    order_delivered_customer_date    datetime64[ns]
    order_estimated_delivery_date    datetime64[ns]
    payment_sequential                      float64
    payment_type                             object
    payment_installments                    float64
    payment_value                           float64
    order_item_id                           float64
    product_id                               object
    seller_id                                object
    shipping_limit_date              datetime64[ns]
    price                                   float64
    freight_value                           float64
    review_id                                object
    review_score                            float64
    review_creation_date             datetime64[ns]
    review_answer_timestamp          datetime64[ns]
    product_category_name                    object
    product_name_length                     float64
    product_description_length              float64
    product_photos_qty                      float64
    product_weight_g                        float64
    product_length_cm                       float64
    product_height_cm                       float64
    product_width_cm                        float64
    product_category_name_english            object
    seller_zip_code_prefix                  float64
    seller_city                              object
    seller_state                             object
    seller_latitude                         float64
    seller_longitude                        float64
    dtype: object



Some columns are misspelt, we will rename them properly


```python
# rename product_name_lenght and product_description_lenght
df.rename(columns = {'product_description_lenght': 'product_description_length', 
          'product_name_lenght': 'product_name_length'}, inplace = True)
```


```python
# check the renamed-dataframe
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 115036 entries, 0 to 118820
    Data columns (total 42 columns):
     #   Column                         Non-Null Count   Dtype         
    ---  ------                         --------------   -----         
     0   customer_id                    115036 non-null  object        
     1   customer_unique_id             115036 non-null  object        
     2   customer_zip_code_prefix       115036 non-null  int64         
     3   customer_city                  115036 non-null  object        
     4   customer_state                 115036 non-null  object        
     5   customer_latitude              115036 non-null  float64       
     6   customer_longitude             115036 non-null  float64       
     7   order_id                       115036 non-null  object        
     8   order_status                   115036 non-null  object        
     9   order_purchase_timestamp       115036 non-null  datetime64[ns]
     10  order_approved_at              115036 non-null  datetime64[ns]
     11  order_delivered_carrier_date   115036 non-null  datetime64[ns]
     12  order_delivered_customer_date  115036 non-null  datetime64[ns]
     13  order_estimated_delivery_date  115036 non-null  datetime64[ns]
     14  payment_sequential             115036 non-null  float64       
     15  payment_type                   115036 non-null  object        
     16  payment_installments           115036 non-null  float64       
     17  payment_value                  115036 non-null  float64       
     18  order_item_id                  115036 non-null  float64       
     19  product_id                     115036 non-null  object        
     20  seller_id                      115036 non-null  object        
     21  shipping_limit_date            115036 non-null  datetime64[ns]
     22  price                          115036 non-null  float64       
     23  freight_value                  115036 non-null  float64       
     24  review_id                      115036 non-null  object        
     25  review_score                   115036 non-null  float64       
     26  review_creation_date           115036 non-null  datetime64[ns]
     27  review_answer_timestamp        115036 non-null  datetime64[ns]
     28  product_category_name          115036 non-null  object        
     29  product_name_length            115036 non-null  float64       
     30  product_description_length     115036 non-null  float64       
     31  product_photos_qty             115036 non-null  float64       
     32  product_weight_g               115036 non-null  float64       
     33  product_length_cm              115036 non-null  float64       
     34  product_height_cm              115036 non-null  float64       
     35  product_width_cm               115036 non-null  float64       
     36  product_category_name_english  115036 non-null  object        
     37  seller_zip_code_prefix         115036 non-null  float64       
     38  seller_city                    115036 non-null  object        
     39  seller_state                   115036 non-null  object        
     40  seller_latitude                115036 non-null  float64       
     41  seller_longitude               115036 non-null  float64       
    dtypes: datetime64[ns](8), float64(19), int64(1), object(14)
    memory usage: 37.7+ MB



```python
# sanity check
df.head()
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
      <th>customer_id</th>
      <th>customer_unique_id</th>
      <th>customer_zip_code_prefix</th>
      <th>customer_city</th>
      <th>customer_state</th>
      <th>customer_latitude</th>
      <th>customer_longitude</th>
      <th>order_id</th>
      <th>order_status</th>
      <th>order_purchase_timestamp</th>
      <th>...</th>
      <th>product_weight_g</th>
      <th>product_length_cm</th>
      <th>product_height_cm</th>
      <th>product_width_cm</th>
      <th>product_category_name_english</th>
      <th>seller_zip_code_prefix</th>
      <th>seller_city</th>
      <th>seller_state</th>
      <th>seller_latitude</th>
      <th>seller_longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06b8999e2fba1a1fbc88172c00ba8bc7</td>
      <td>861eff4711a542e4b93843c6dd7febb0</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>00e7ee1b050b8499577073aeb2a297a1</td>
      <td>delivered</td>
      <td>2017-05-16 15:05:35</td>
      <td>...</td>
      <td>8683.0</td>
      <td>54.0</td>
      <td>64.0</td>
      <td>31.0</td>
      <td>office_furniture</td>
      <td>8577.0</td>
      <td>itaquaquecetuba</td>
      <td>SP</td>
      <td>-23.482623</td>
      <td>-46.374490</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5dca924cc99eea2dc5ba40d11ec5dd0f</td>
      <td>2761fee7f378f0a8d7682d8a3fa07ab1</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>95261f608a64bbbe760a212b4d48a4ee</td>
      <td>delivered</td>
      <td>2018-06-15 20:07:13</td>
      <td>...</td>
      <td>1383.0</td>
      <td>50.0</td>
      <td>10.0</td>
      <td>40.0</td>
      <td>bed_bath_table</td>
      <td>14940.0</td>
      <td>ibitinga</td>
      <td>SP</td>
      <td>-21.766477</td>
      <td>-48.831547</td>
    </tr>
    <tr>
      <th>2</th>
      <td>661897d4968f1b59bfff74c7eb2eb4fc</td>
      <td>d06a495406b79cb8203ea21cc0942f8c</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>9444fa2ab50a3f5af63b48be297eda24</td>
      <td>delivered</td>
      <td>2017-09-09 15:40:00</td>
      <td>...</td>
      <td>1200.0</td>
      <td>47.0</td>
      <td>7.0</td>
      <td>26.0</td>
      <td>toys</td>
      <td>89204.0</td>
      <td>joinville</td>
      <td>SC</td>
      <td>-26.283149</td>
      <td>-48.851285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>702b62324327ccba20f1be3465426437</td>
      <td>8b3d988f330c1d1c3332ccd440c147b7</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>dceb8e88274c6f42a88a76ed979eb817</td>
      <td>delivered</td>
      <td>2018-03-26 12:04:55</td>
      <td>...</td>
      <td>567.0</td>
      <td>19.0</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>auto</td>
      <td>4243.0</td>
      <td>sao paulo</td>
      <td>SP</td>
      <td>-23.626269</td>
      <td>-46.586534</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bdf997bae7ca819b0415f5174d6b4302</td>
      <td>866755e25db620f8d7e81b351a15bb2f</td>
      <td>14409</td>
      <td>franca</td>
      <td>SP</td>
      <td>-20.509897</td>
      <td>-47.397866</td>
      <td>6ee1cea1b2edcc713f83ebfbccbc57f9</td>
      <td>delivered</td>
      <td>2018-07-29 20:39:20</td>
      <td>...</td>
      <td>300.0</td>
      <td>20.0</td>
      <td>7.0</td>
      <td>15.0</td>
      <td>telephony</td>
      <td>1212.0</td>
      <td>sao paulo</td>
      <td>SP</td>
      <td>-23.537511</td>
      <td>-46.637057</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>



We are satisfied with our new dataframe, we will export as csv.


```python
# export and save as a csv file
df.to_csv('cleaned_df.csv', index = False)
```

### Conclusion

Here, we have successfully import all the datasets we need for the project, merge them and cleaned the data by removing null values or forward fill the values, duplicates were also removed. The result of **Notebook #2 for feature engineering.
