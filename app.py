import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle

#TITLE
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price Prediction')

st.image('https://png.pngtree.com/thumb_back/fh260/background/20220729/pngtree-residential-houses-and-a-yen-yuan-money-bag-buyingfair-price-city-municipal-budget-property-real-estate-valuation-mortgage-loan-calculation-of-construction-and-repair-expenses-photo-image_32959858.jpg')

st.header('A model of housing prices to predict median house values in California', divider = True)
st.header('''User must enter given values to predict Price:''')

st.sidebar.title('Select House Features 🏠')
st.sidebar.image('https://png.pngtree.com/thumb_back/fh260/background/20250603/pngtree-modern-suburban-house-exterior-at-sunset-image_17383480.jpg')

# read_data
temp_df = pd.read_csv('california.csv')
random.seed(12)
all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min', 'max'])
    
    var = st.sidebar.slider(f'Select {i} Value', int(min_value), int(max_value),
                            random.randint(int(min_value),int(max_value)))
    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])


with open('House_Price_Pred_Ridge_Model.pkl', 'rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]



import time

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))

progress_bar = st.progress(0)
place_holder = st.empty()
place_holder.write('Predicting Price')

place=st.empty()
place.image('https://i.pinimg.com/originals/dc/cc/84/dccc846959dffafa30a836dfacf9bab9.gif', width = 200)

if price>0:
    
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
        
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    place_holder.empty()
    place.empty()
    st.success(body)
    
else:
    body = 'Invalid house features values'
    st.warning(body)



    