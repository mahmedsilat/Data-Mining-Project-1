import pandas as pd
data = pd.read_csv("ks-projects-201801.csv")  # or pd.read_excel, pd.read_json, etc.


indexNames = data[ data['state'] == 'cancelled' ].index
data.drop(indexNames , inplace=True)


# Calculate the correlation between goal amount and success rate
# correlation = data[['usd_goal_real', 'success']].corr().loc['usd_goal_real', 'success']

# print(f"The correlation between goal amount and success rate is: {correlation}")



print(data.head())       # View the first few rows
# print(data.info())       # Get an overview of data types and missing values
# print(data.describe())   # Summary statistics for numerical columns
# data.duplicated().sum()  # Count duplicate rows
# print(data.isnull().sum())  # Count missing values in each column


data['name'] = data['name'].fillna('Unknown')
data = data.dropna(subset=['usd pledged'])
# Save the edited DataFrame back to the original file
data.to_csv("ks-projects-201801.csv", index=False)  # index=False to avoid saving the index as a separate column
print(data.head())   
# data['ID'] = data['ID'].astype('int')  # Convert ID to integer
# data['goal'] = data['goal'].astype('float')  # Convert goal to float
# data['backers'] = data['backers'].astype('int')  # Convert backers to integer
# data['deadline'] = pd.to_datetime(data['launched'], format='%d/%m/%Y %H:%M')# Convert launched to datetime
# data['deadline'] = pd.to_datetime(data['deadline'], format='%d/%m/%Y')    # Convert deadline to datetime
# data.to_csv("ks-projects-201801.csv", index=False)  # index=False to avoid saving the index as a separate column
