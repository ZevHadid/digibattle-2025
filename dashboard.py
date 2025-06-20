import pandas as pd
import streamlit as st
import seaborn as sns
import pycountry
import pycountry_convert as pc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import plotly.express as px
    
# st.set_page_config(layout="wide")
df = pd.read_csv('sales_ecommerce_filtered.csv')

st.title("Platform E-commerce Data Analysis")

# Sidebar
st.sidebar.title("Feature Options")
products = df['ProductName'].unique()
countries = df['Country'].unique()

selected_product = st.sidebar.selectbox("Select Product", products, key="product_selectbox_1")
selected_country = st.sidebar.selectbox("Select Country", countries, key="country_selectbox")



st.write("### Key Sales Metrics")

# Total sales
total_sales = (df['Price'] * df['Quantity']).sum()

# Total transactions
total_transactions = df['TransactionNo'].nunique()

# Best-selling product
product_sales_all = (
    df.groupby('ProductName')
    .apply(lambda x: (x['Price'] * x['Quantity']).sum())
    .reset_index(name='TotalSales')
)
best_product = product_sales_all.sort_values('TotalSales', ascending=False).iloc[0]

# Top revenue country
country_sales_all = (
    df.groupby('Country')
    .apply(lambda x: (x['Price'] * x['Quantity']).sum())
    .reset_index(name='TotalSales')
)
top_country = country_sales_all.sort_values('TotalSales', ascending=False).iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"<div style='font-size:14px'>Total Sales</div>"
        f"<div style='font-size:18px; font-weight:bold;'>${total_sales:,.2f}</div>",
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"<div style='font-size:14px'>Total Transactions</div>"
        f"<div style='font-size:18px; font-weight:bold;'>{total_transactions:,}</div>",
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"<div style='font-size:14px'>Best-Selling Product</div>"
        f"<div style='font-size:16px; font-weight:bold;'>{best_product['ProductName']}</div>"
        f"<div style='font-size:14px;'>${best_product['TotalSales']:,.2f}</div>",
        unsafe_allow_html=True
    )
with col4:
    st.markdown(
        f"<div style='font-size:14px'>Top Revenue Country</div>"
        f"<div style='font-size:16px; font-weight:bold;'>{top_country['Country']}</div>"
        f"<div style='font-size:14px;'>${top_country['TotalSales']:,.2f}</div>",
        unsafe_allow_html=True
    )

st.write("### Data Overview")
st.write(df.head())

st.write("### Basic Statistics")
st.write(df.describe())

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Filter the data based on user selection
filtered_df = df[(df['ProductName'] == selected_product) & (df['Country'] == selected_country)].copy()

# Create a new column 'TotalSales'
filtered_df['TotalSales'] = filtered_df['Price'] * filtered_df['Quantity']

# Group by 'Date' and sum 'TotalSales'
trend_df = filtered_df.groupby('Date', as_index=False)['TotalSales'].sum()



# Line Plot
st.write("### Trend Visualization")

fig2, ax2 = plt.subplots()
ax2.plot(trend_df['Date'], trend_df['TotalSales'], marker='o')
ax2.set_title(f"Sales Trend for {selected_product} in {selected_country}")
ax2.set_xlabel("Date")
ax2.set_ylabel("Total Sales")
plt.xticks(rotation=45)
st.pyplot(fig2)

# Aggregate total sales per product for the selected country
country_df = df[df['Country'] == selected_country]
product_sales = (
    country_df.groupby('ProductName')
    .apply(lambda x: (x['Price'] * x['Quantity']).sum())
    .reset_index()
)
product_sales.columns = ['ProductName', 'TotalSales']

# Get top 3 products
st.write("### Top 3 Products by Total Sales")
top3_products = product_sales.sort_values('TotalSales', ascending=False).head(3)

# Bar plot
fig3, ax3 = plt.subplots()
sns.barplot(data=top3_products, x='ProductName', y='TotalSales', ax=ax3)
ax3.set_title(f"Top 3 Products in {selected_country}")
ax3.set_xlabel("Product")
ax3.set_ylabel("Total Sales")
plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')
st.pyplot(fig3)

st.write("### Top Customers by Total Spend")

# Calculate total spend per customer
customer_sales = (
    df.groupby('CustomerNo')
    .apply(lambda x: (x['Price'] * x['Quantity']).sum())
    .reset_index(name='TotalSpent')
    .sort_values('TotalSpent', ascending=False)
    .dropna(subset=['CustomerNo'])
)

# Get top 10 customers
top_customers = customer_sales.head(10)

# Pie chart
fig_pie = px.pie(
    top_customers,
    names='CustomerNo',
    values='TotalSpent',
    hover_data={'TotalSpent': ':.2f'},
    labels={'TotalSpent': 'Total Spent'}
)
fig_pie.update_traces(textinfo='percent+label', pull=[0.05]*10)
st.plotly_chart(fig_pie, use_container_width=True)

# Show table of top customers with total spent
st.dataframe(top_customers.reset_index(drop=True))

st.write("### Country Map: Top Buyers of Selected Product")

# Prepare sales data for the selected product
product_country_sales = (
    df[df['ProductName'] == selected_product]
    .groupby('Country')
    .apply(lambda x: (x['Price'] * x['Quantity']).sum())
    .reset_index()
)
product_country_sales.columns = ['Country', 'TotalSales']

# Calculate percentage of total sales per country for the selected product
total_sales = product_country_sales['TotalSales'].sum()
product_country_sales['Percentage'] = (product_country_sales['TotalSales'] / total_sales) * 100

def get_iso3(country_name):
    try:
        mapping = {
            'EIRE': 'IRL',
            'USA': 'USA',
            'RSA': 'ZAF',
            'Channel Islands': 'GGY',
            'European Community': None,
            'Unspecified': None,
            'UK': 'GBR',
            'United Kingdom': 'GBR'
        }
        if country_name in mapping:
            return mapping[country_name]
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return None

product_country_sales['iso_alpha'] = product_country_sales['Country'].apply(get_iso3)
product_country_sales = product_country_sales.dropna(subset=['iso_alpha'])

# Set a wider color range for the percentage (0 to 100)
fig_map = px.choropleth(
    product_country_sales,
    color="Percentage",
    locations="iso_alpha",
    hover_data={"Percentage": ':.2f', "TotalSales": True},
    color_continuous_scale=px.colors.sequential.OrRd,
    range_color=(0, 100),
    labels={'Percentage': '% of Sales'},
    title=f"Country Share of Sales for '{selected_product}'"
)
fig_map.update_geos(showcoastlines=True, showland=True, fitbounds="locations")
fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})

st.plotly_chart(fig_map, use_container_width=True)

# Correlation heatmap for numerical data
st.write("### Correlation Heatmap")
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)  
st.write("### Predict Customer Churn (Likelihood of Repeat Purchase)")

# Prepare data for churn prediction
# Churn: Whether a customer made a repeat purchase (1) or not (0)
churn_df = df.copy()
churn_df['TotalSales'] = churn_df['Price'] * churn_df['Quantity']

# For each customer, check if they have more than one transaction
customer_purchase_counts = churn_df.groupby('CustomerNo')['TransactionNo'].nunique()
churn_df['RepeatPurchase'] = churn_df['CustomerNo'].map(lambda x: 1 if customer_purchase_counts.get(x, 0) > 1 else 0)

# Drop rows with missing CustomerNo
churn_df = churn_df.dropna(subset=['CustomerNo'])

# Encode categorical variables
churn_df['ProductName_enc'] = churn_df['ProductName'].astype('category').cat.codes
churn_df['Country_enc'] = churn_df['Country'].astype('category').cat.codes

# Features and target
X = churn_df[['Year', 'Month', 'Day', 'ProductName_enc', 'Country_enc', 'Price', 'Quantity']]
y = churn_df['RepeatPurchase']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model (RandomForestClassifier for classification)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

st.write(f"**Accuracy:** {acc:.2f}")
st.write(f"**ROC AUC Score:** {roc_auc:.2f}")

# User input for prediction
st.write("#### Predict Repeat Purchase Likelihood for a Transaction")
future_products = df['ProductName'].unique()
future_countries = df['Country'].unique()

future_product = st.selectbox("Select Product", future_products, key="product_selectbox_churn")
future_country = st.selectbox("Select Country", future_countries, key="country_selectbox_churn")
future_price = st.number_input("Enter Price", min_value=0.0, value=float(df['Price'].mean()))
future_quantity = st.number_input("Enter Quantity", min_value=1, value=1)

# Only allow dates after the latest date in the data
latest_date = df['Date'].max()
min_date = latest_date + pd.Timedelta(days=1)
future_date = st.date_input("Select Date", min_value=min_date, value=min_date)

future_year = future_date.year
future_month = future_date.month
future_day = future_date.day

# Encode product and country
product_enc = churn_df[churn_df['ProductName'] == future_product]['ProductName_enc'].iloc[0]
country_enc = churn_df[churn_df['Country'] == future_country]['Country_enc'].iloc[0]

input_features = [[future_year, future_month, future_day, product_enc, country_enc, future_price, future_quantity]]
repeat_purchase_prob = clf.predict_proba(input_features)[0][1]

st.write(f"**Predicted Probability of Repeat Purchase:** {repeat_purchase_prob*100:.1f}%")

df.info()