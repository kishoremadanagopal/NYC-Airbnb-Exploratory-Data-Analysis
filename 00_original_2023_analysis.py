import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv("AB_NYC_2019.csv")

print(df.head())  # Display the first few rows to understand the structure of the data
print(df.info())  # Check data types and missing values

# Let's check for missing values in each column
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)
duplicate_rows = df[df.duplicated()]
print("Duplicate Rows:\n", duplicate_rows)


# Assigning variable
rental_price = df.loc[:, 'price']
#Calculate rental price statistics
mean_rent = np.mean(rental_price)
print('The average rental price is: $', round(mean_rent, 2))

max_rent = np.max(rental_price)
print('The maximum rental price is: $', round(max_rent, 2))

min_rent = np.min(rental_price)
print('The minimum rental price is: $', round(min_rent, 2))

rent_std = np.std(rental_price)
print('The standard deviation of rental prices is: $', round(rent_std, 2))

median_rent = np.median(rental_price)
print('The median rental price is: $', round(median_rent, 2))

# number of neighbourhood groups present in newyork

region = df.loc[:, 'neighbourhood_group']
unique_regions = pd.unique(region)
print('Number of distinct regions:', unique_regions)

# Types of rooms available in newyork
categories = pd.unique(df.room_type)
print('The property categories registered in New York City are: ', categories)


# Graph 1 - Most Expensive Rentals by Region in New York
# Group the data by region type and calculate the maximum rental prices
max_rental_by_region = df.loc[:, ['neighbourhood_group', 'price']].groupby('neighbourhood_group').max()
# Convert the values to integers
max_rental_by_region = max_rental_by_region.astype(int)
# Reset the index to make 'neighbourhood_group' a column again
max_rental_by_region.reset_index(inplace=True)
# Rename the columns for better understanding
max_rental_by_region.columns = ['Region', 'Maximum Rental']
# Seaborn barplot for maximum rental prices by region
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Region', y='Maximum Rental', hue='Region', data=max_rental_by_region,palette='Blues_d')
plt.title('Most Expensive Rentals by Region in New York')
plt.xlabel('Region')
plt.ylabel('Maximum Rental')
# Add data labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 7), textcoords='offset points')
plt.show()

# Graph - 2 Average Rental Prices by Region in New York
# Group the data by region type and calculate the average rental prices
average_rental_by_region = df.loc[:, ['neighbourhood_group', 'price']].groupby('neighbourhood_group').mean()
# Convert the values to integers
average_rental_by_region = average_rental_by_region.astype(int)
# Reset the index to make 'neighbourhood_group' a column again
average_rental_by_region.reset_index(inplace=True)
# Rename the columns for better understanding
average_rental_by_region.columns = ['Region', 'Average Rental']
# Seaborn barplot for average rental prices by region
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Region', y='Average Rental', data=average_rental_by_region, palette='Blues_d')
plt.title('Average Rental Prices by Region in New York')
plt.xlabel('Region')
plt.ylabel('Average Rental')
# Add data labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 7), textcoords='offset points')
plt.show()

# Graph - 3 Group the data by region and room type and calculate the maximum rental value
max_value_region_type = df.loc[:, ['neighbourhood_group', 'room_type', 'price']].groupby(['neighbourhood_group', 'room_type']).max().reset_index()
# Convert the values to integers
max_value_region_type['price'] = max_value_region_type['price'].astype(int)
# Rename the columns for better understanding
max_value_region_type.columns = ['Region', 'Type', 'Maximum Value']
fig = px.bar(max_value_region_type, x='Maximum Value', y='Region', color='Type', orientation='h',
             labels={'Maximum Value':'Maximum Rental Price', 'Region':'Region', 'Type':'Room Type'},
             title='Maximum Rental Price by Region and Room Type in New York City')
fig.update_traces(texttemplate='%{x:.0f}', textposition='inside')
fig.show()


# Graph - 4 - Own - Assuming 'data' is your DataFrame containing columns 'room_type', 'neighbourhood_group', and 'number_of_reviews'
# Grouping to count occurrences of room types within each neighbourhood group
count_room_type = df.groupby('neighbourhood_group')['room_type'].value_counts().unstack().fillna(0)
# Average number of reviews per neighbourhood group
average_reviews_by_group = df.groupby('neighbourhood_group')['number_of_reviews'].mean()
# Average total reviews across all neighbourhood groups
average_total_reviews = df['number_of_reviews'].mean()
# Defining colors for each room type
colors = ['skyblue', 'salmon', 'lightgreen']  # Add more colors as needed
# Creating the dual-axis plot
fig, ax1 = plt.subplots(figsize=(10, 6))
# Bar plot for room type counts with different colors
count_room_type.plot(kind='bar', ax=ax1, color=colors)
ax1.set_xlabel('Neighbourhood Group')
ax1.set_ylabel('Room Type Count')
# Extracting unique values from the variable
categories = pd.unique(df.room_type)
print('The property categories registered in New York City are: ', categories)
# Line plot for average reviews by neighbourhood group on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(average_reviews_by_group.index, average_reviews_by_group.values, color='purple', marker='o', linestyle='--', label='Average Reviews')
ax2.axhline(y=average_total_reviews, color='orange', linestyle='--', label='Overall Average Reviews')
ax2.set_ylabel('Average Reviews & Overall Average', color='purple')
ax2.legend(loc='upper right')
# Legends for room types
legend_labels = count_room_type.columns.tolist()
ax1.legend(legend_labels, loc='upper left')
plt.title('Room Type Count and Average Reviews by Neighbourhood Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Graph 5 - Pivot the data to get availability and price per neighbourhood group
availability_pivot = df.pivot_table(values='availability_365', index='neighbourhood_group', aggfunc='mean')
price_pivot = df.pivot_table(values='price', index='neighbourhood_group', aggfunc='mean')
# Create a heatmap combining availability and price using Seaborn
plt.figure(figsize=(12, 8))
# Heatmap for availability
plt.subplot(1, 2, 1)
sns.heatmap(availability_pivot, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=0.5)
plt.title('Average Availability by Neighbourhood Group')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Availability (Average Days)')
# Heatmap for price
plt.subplot(1, 2, 2)
sns.heatmap(price_pivot, cmap='YlOrRd', annot=True, fmt=".0f", linewidths=0.5)
plt.title('Average Price by Neighbourhood Group')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Price (Average)')
plt.tight_layout()
plt.show()

# Graph 6 - Seaborn histogram with x-axis limit, binwidth of 5, x-ticks and y-ticks for number of reviews
plt.figure(figsize=(10, 6))
sns.histplot(df[df['number_of_reviews'] <= 100], x='number_of_reviews', binwidth=5, kde=False, color='blue')
plt.title('Distribution of Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, 101, 5))
plt.yticks(np.arange(0, 24001, 2000))
plt.show()

#Graph 7 - Seaborn histogram for hosts with up to 10 properties with data labels
plt.figure(figsize=(10, 6))
ax = sns.histplot(df[df['calculated_host_listings_count'] <= 10], x='calculated_host_listings_count', binwidth=1, kde=False, color='blue')
plt.title('Distribution of Properties per Host')
plt.xlabel('Number of Properties')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, 11, 1))

# Add data labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.show()