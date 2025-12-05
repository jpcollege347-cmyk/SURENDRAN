import pandas as pd
import streamlit as st
import io
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set page configuration
st.set_page_config(layout="wide")
st.title('Amazon Sales Data Analysis')

# Function to load data (using Streamlit's caching for efficiency)
@st.cache_data
def load_data():
    df = pd.read_csv('/content/Amazon.csv')
    return df

df = load_data()
df_cleaned = df.copy() # Create a copy for cleaning operations

st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', ['Data Preview', 'Data Cleaning', 'Exploratory Data Analysis', 'Insights', 'Export Data'])

# 1. Data Preview Section
if page == 'Data Preview':
    st.header('1. Data Preview')

    st.subheader('First 5 Rows of the Dataset:')
    st.dataframe(df_cleaned.head())

    st.subheader('Dataset Shape:')
    st.write(f"Rows: {df_cleaned.shape[0]}, Columns: {df_cleaned.shape[1]}")

    st.subheader('Column Names:')
    st.write(df_cleaned.columns.tolist())

    st.subheader('Data Types and Non-Null Values:')
    buffer = io.StringIO()
    df_cleaned.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader('Missing Values Count:')
    st.dataframe(df_cleaned.isnull().sum().rename('Missing Values'))

# 2. Data Cleaning Section
elif page == 'Data Cleaning':
    st.header('2. Data Cleaning')

    # Identify and display the count of duplicate rows
    duplicate_rows = df_cleaned.duplicated().sum()
    st.subheader('Duplicate Rows:')
    st.write(f"Number of duplicate rows found: {duplicate_rows}")

    # Remove duplicate rows from the DataFrame
    if duplicate_rows > 0:
        df_cleaned.drop_duplicates(inplace=True)
        st.write('Duplicate rows removed.')
    else:
        st.write('No duplicate rows found.')

    # Identify and display the count of missing values for each column after duplicates removal
    st.subheader('Missing Values After Duplicate Removal:')
    missing_values_after_duplicates = df_cleaned.isnull().sum()
    st.dataframe(missing_values_after_duplicates.rename('Missing Values'))

    # Handle missing values (imputation)
    st.subheader('Handling Missing Values:')
    for column in df_cleaned.columns:
        if df_cleaned[column].isnull().any():
            if df_cleaned[column].dtype == 'object':  # Categorical column
                mode_value = df_cleaned[column].mode()[0]  # Get the first mode if multiple exist
                df_cleaned[column].fillna(mode_value, inplace=True)
                st.write(f"Filled missing values in '{column}' with mode: {mode_value}")
            elif pd.api.types.is_numeric_dtype(df_cleaned[column]):  # Numerical column
                median_value = df_cleaned[column].median()
                df_cleaned[column].fillna(median_value, inplace=True)
                st.write(f"Filled missing values in '{column}' with median: {median_value}")
            else:
                st.write(f"Column '{column}' has missing values but is neither categorical nor numeric. No imputation performed.")

    # Convert 'OrderDate' column to datetime
    st.subheader('Data Type Conversion:')
    df_cleaned['OrderDate'] = pd.to_datetime(df_cleaned['OrderDate'])
    st.write("Converted 'OrderDate' to datetime format.")

    # Convert numeric columns like 'Sales' and 'Profit' if they exist and are not already numeric
    st.subheader('Numerical Column Check:')
    if 'Sales' in df_cleaned.columns:
        df_cleaned['Sales'] = df_cleaned['Sales'].replace({r'[$,]': ''}, regex=True).astype(float)
        st.write("Converted 'Sales' to numeric (float) format.")
    else:
        st.write("Column 'Sales' not found. 'TotalAmount' is already numeric (float64).")

    if 'Profit' in df_cleaned.columns:
        df_cleaned['Profit'] = df_cleaned['Profit'].replace({r'[$,]': ''}, regex=True).astype(float)
        st.write("Converted 'Profit' to numeric (float) format.")
    else:
        st.write("Column 'Profit' not found.")

    # Standardize category/text fields (example for 'Category' and 'PaymentMethod')
    st.subheader('Text Standardization:')
    if 'Category' in df_cleaned.columns and df_cleaned['Category'].dtype == 'object':
        df_cleaned['Category'] = df_cleaned['Category'].str.strip().str.lower()
        st.write("Standardized 'Category' column (stripped whitespace, converted to lowercase).")
    if 'PaymentMethod' in df_cleaned.columns and df_cleaned['PaymentMethod'].dtype == 'object':
        df_cleaned['PaymentMethod'] = df_cleaned['PaymentMethod'].str.strip().str.lower()
        st.write("Standardized 'PaymentMethod' column (stripped whitespace, converted to lowercase).")


    # Display total number of missing values and final shape after cleaning
    st.subheader('Cleaning Summary:')
    total_missing_after_cleaning = df_cleaned.isnull().sum().sum()
    st.write(f"Total missing values after cleaning: {total_missing_after_cleaning}")
    st.write(f"Final DataFrame Shape: Rows: {df_cleaned.shape[0]}, Columns: {df_cleaned.shape[1]}")
    st.dataframe(df_cleaned.head())

# 3. Exploratory Data Analysis (EDA) Section
elif page == 'Exploratory Data Analysis':
    st.header('3. Exploratory Data Analysis (EDA)')

    # 1. Display descriptive statistics for numerical columns
    st.subheader('Descriptive Statistics for Numerical Columns:')
    st.dataframe(df_cleaned.describe())

    # 2. Identify and display the count of unique values for categorical columns
    st.subheader('Unique Value Counts for Categorical Columns:')
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype == 'object':  # Categorical column
            unique_count = df_cleaned[column].nunique()
            st.write(f"'{column}': {unique_count} unique values")

    # 3. Create an interactive bar chart showing total sales by product category
    if 'Category' in df_cleaned.columns and 'TotalAmount' in df_cleaned.columns:
        st.subheader('Total Sales by Product Category:')
        sales_by_category = df_cleaned.groupby('Category')['TotalAmount'].sum().reset_index()
        chart_category_sales = alt.Chart(sales_by_category).mark_bar().encode(
            x=alt.X('Category', sort='-y', title='Product Category'),
            y=alt.Y('TotalAmount', title='Total Sales'),
            tooltip=['Category', 'TotalAmount']
        ).properties(title='Total Sales by Product Category').interactive()
        st.altair_chart(chart_category_sales, use_container_width=True)
    else:
        st.write("Cannot generate 'Sales by Product Category' chart: 'Category' or 'TotalAmount' column not found.")

    # 4. Analyze total sales over time (by month)
    if 'OrderDate' in df_cleaned.columns and 'TotalAmount' in df_cleaned.columns:
        st.subheader('Total Sales Over Time (Monthly):')
        df_cleaned['YearMonth'] = df_cleaned['OrderDate'].dt.to_period('M').astype(str)
        sales_over_time = df_cleaned.groupby('YearMonth')['TotalAmount'].sum().reset_index()
        chart_time_series = alt.Chart(sales_over_time).mark_line(point=True).encode(
            x=alt.X('YearMonth', sort=None, title='Month'),
            y=alt.Y('TotalAmount', title='Total Sales'),
            tooltip=['YearMonth', 'TotalAmount']
        ).properties(title='Total Sales Over Time (Monthly)').interactive()
        st.altair_chart(chart_time_series, use_container_width=True)
    else:
        st.write("Cannot generate 'Sales Over Time' chart: 'OrderDate' or 'TotalAmount' column not found.")

    # 5. Identify the top 10 products by total sales
    if 'ProductName' in df_cleaned.columns and 'TotalAmount' in df_cleaned.columns:
        st.subheader('Top 10 Products by Total Sales:')
        top_products_sales = df_cleaned.groupby('ProductName')['TotalAmount'].sum().nlargest(10).reset_index()
        chart_top_products = alt.Chart(top_products_sales).mark_bar().encode(
            x=alt.X('ProductName', sort='-y', title='Product Name'),
            y=alt.Y('TotalAmount', title='Total Sales'),
            tooltip=['ProductName', 'TotalAmount']
        ).properties(title='Top 10 Products by Total Sales').interactive()
        st.altair_chart(chart_top_products, use_container_width=True)
    else:
        st.write("Cannot generate 'Top 10 Products by Sales' chart: 'ProductName' or 'TotalAmount' column not found.")

    # 6. Analyze total sales by payment method
    if 'PaymentMethod' in df_cleaned.columns and 'TotalAmount' in df_cleaned.columns:
        st.subheader('Total Sales by Payment Method:')
        sales_by_payment = df_cleaned.groupby('PaymentMethod')['TotalAmount'].sum().reset_index()
        chart_payment_sales = alt.Chart(sales_by_payment).mark_bar().encode(
            x=alt.X('PaymentMethod', sort='-y', title='Payment Method'),
            y=alt.Y('TotalAmount', title='Total Sales'),
            tooltip=['PaymentMethod', 'TotalAmount']
        ).properties(title='Total Sales by Payment Method').interactive()
        st.altair_chart(chart_payment_sales, use_container_width=True)
    else:
        st.write("Cannot generate 'Sales by Payment Method' chart: 'PaymentMethod' or 'TotalAmount' column not found.")

    # 7. Generate a correlation heatmap for numerical columns
    st.subheader('Correlation Heatmap of Numerical Columns:')
    numerical_df = df_cleaned.select_dtypes(include=['number'])
    if not numerical_df.empty:
        corr_matrix = numerical_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title('Correlation Matrix of Numerical Columns')
        st.pyplot(fig)
    else:
        st.write("No numerical columns found to generate a correlation heatmap.")

# 4. Insights Section
elif page == 'Insights':
    st.header('4. Insights')
    st.subheader('Automatically Generated Insights:')

    insights = []

    # Insight 1: Top performing categories
    if 'Category' in df_cleaned.columns and 'TotalAmount' in df_cleaned.columns:
        sales_by_category = df_cleaned.groupby('Category')['TotalAmount'].sum().nlargest(1)
        if not sales_by_category.empty:
            top_category = sales_by_category.index[0]
            top_category_sales = sales_by_category.values[0]
            insights.append(f"- The top-performing product category is **{top_category}** with total sales of **${top_category_sales:,.2f}**.")

    # Insight 2: Top performing products
    if 'ProductName' in df_cleaned.columns and 'TotalAmount' in df_cleaned.columns:
        sales_by_product = df_cleaned.groupby('ProductName')['TotalAmount'].sum().nlargest(1)
        if not sales_by_product.empty:
            top_product = sales_by_product.index[0]
            top_product_sales = sales_by_product.values[0]
            insights.append(f"- The best-selling product is **{top_product}** generating **${top_product_sales:,.2f}** in sales.")

    # Insight 3: Sales trend over time (simplified for quick insight)
    if 'OrderDate' in df_cleaned.columns and 'TotalAmount' in df_cleaned.columns:
        df_cleaned['YearMonth'] = df_cleaned['OrderDate'].dt.to_period('M').astype(str)
        sales_over_time = df_cleaned.groupby('YearMonth')['TotalAmount'].sum()
        if not sales_over_time.empty:
            if sales_over_time.iloc[-1] > sales_over_time.iloc[0]:
                insights.append(f"- Overall sales show an **increasing trend** over the observed period.")
            else:
                insights.append(f"- Overall sales show a **stable or slightly decreasing trend** over the observed period.")

    # Insight 4: Most popular payment method
    if 'PaymentMethod' in df_cleaned.columns:
        popular_payment = df_cleaned['PaymentMethod'].mode()[0]
        insights.append(f"- The most frequently used payment method by customers is **{popular_payment.capitalize()}**.")

    # Insight 5: Correlation (simple interpretation if strong correlation exists)
    numerical_df = df_cleaned.select_dtypes(include=['number'])
    if not numerical_df.empty and len(numerical_df.columns) > 1:
        corr_matrix = numerical_df.corr()
        # Check for strong positive correlation between TotalAmount and Quantity if both exist
        if 'TotalAmount' in corr_matrix.columns and 'Quantity' in corr_matrix.columns:
            correlation = corr_matrix.loc['TotalAmount', 'Quantity']
            if correlation > 0.7:
                insights.append(f"- There is a **strong positive correlation ({correlation:.2f})** between `Quantity` and `TotalAmount`, indicating that higher quantities sold directly lead to higher total sales.")

    if insights:
        for insight in insights:
            st.write(insight)
    else:
        st.write("No specific insights generated due to data limitations or absence of relevant columns.")

# 5. Export Options Section
elif page == 'Export Data':
    st.header('5. Export Data')

    # Download Cleaned Data
    st.subheader('Download Cleaned Data')
    csv_cleaned = df_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned CSV",
        data=csv_cleaned,
        file_name='amazon_cleaned_data.csv',
        mime='text/csv',
    )

    # Download Summary Report (a simple text report for demonstration)
    st.subheader('Download Summary Report')
    report_buffer = io.StringIO()
    report_buffer.write('Amazon Sales Data Analysis Summary Report\n\n')
    report_buffer.write(f'Date Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    report_buffer.write('--- Data Overview ---\n')
    report_buffer.write(f"Original Rows: {df.shape[0]}, Original Columns: {df.shape[1]}\n")
    report_buffer.write(f"Cleaned Rows: {df_cleaned.shape[0]}, Cleaned Columns: {df_cleaned.shape[1]}\n")
    report_buffer.write(f"Duplicates Removed: {df.shape[0] - df_cleaned.shape[0]}\n")
    report_buffer.write(f"Total Missing Values After Cleaning: {df_cleaned.isnull().sum().sum()}\n\n")

    report_buffer.write('--- Key Insights ---\n')
    if 'insights' in locals(): # Check if insights were generated
        for insight in insights:
            report_buffer.write(f'{insight}\n')
    else:
        report_buffer.write('No specific insights were generated for this run.\n')

    report_text = report_buffer.getvalue()
    b64 = base64.b64encode(report_text.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/txt;base64,{b64}" download="amazon_summary_report.txt">Download Summary Report</a>'
    st.markdown(href, unsafe_allow_html=True)
