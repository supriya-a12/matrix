import pandas as pd
import numpy as np

# Question 1: Matrix Generation
def generate_car_matrix(data):
    # Pivot the DataFrame to create the matrix
    car_matrix = pd.pivot_table(data, values='car', index='id_1', columns='id_2', fill_value=0)
    
    # Set diagonal values to 0
    np.fill_diagonal(car_matrix.values, 0)
    
    return car_matrix

# Question 2: Car Type Count Calculation
def get_type_count(data):
    # Add a new categorical column 'car_type' based on values of the 'car' column
    data['car_type'] = pd.cut(data['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=['low', 'medium', 'high'])
    
    # Calculate the count of occurrences for each 'car_type' category
    type_count = data['car_type'].value_counts().to_dict()
    
    # Sort the dictionary alphabetically based on keys
    type_count = dict(sorted(type_count.items()))
    
    return type_count

# Question 3: Bus Count Index Retrieval
def get_bus_indexes(data):
    # Identify indices where 'bus' values are greater than twice the mean
    bus_mean = data['bus'].mean()
    bus_indexes = data[data['bus'] > 2 * bus_mean].index.tolist()
    
    # Sort indices in ascending order
    bus_indexes.sort()
    
    return bus_indexes

# Question 4: Route Filtering
def filter_routes(data):
    # Group by 'route' and filter based on the average of 'truck' column
    filtered_routes = data.groupby('route')['truck'].mean()
    filtered_routes = filtered_routes[filtered_routes > 7].index.tolist()
    
    # Sort the list of routes
    filtered_routes.sort()
    
    return filtered_routes

# Question 5: Matrix Value Modification
def multiply_matrix(car_matrix):
    # Modify values based on the given logic
    modified_matrix = car_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    
    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)
    
    return modified_matrix

# Question 6: Time Check
def time_check(data):
    # Combine 'startDay' and 'startTime' to create a 'start_timestamp' column
    data['start_timestamp'] = pd.to_datetime(data['startDay'] + ' ' + data['startTime'])
    
    # Group by unique (id, id_2) pairs and check completeness of timestamps
    result = data.groupby(['id', 'id_2']).apply(lambda group: group['start_timestamp'].min() != group['start_timestamp'].max())
    
    return result

# Example usage:
# Load the dataset
dataset1 = pd.read_csv('dataset-1.csv')
dataset2 = pd.read_csv('dataset-2.csv')

# Task 1 Question 1
result1 = generate_car_matrix(dataset1)
print("Task 1 Question 1 Result:")
print(result1)

# Task 1 Question 2
result2 = get_type_count(dataset1)
print("\nTask 1 Question 2 Result:")
print(result2)

# Task 1 Question 3
result3 = get_bus_indexes(dataset1)
print("\nTask 1 Question 3 Result:")
print(result3)

# Task 1 Question 4
result4 = filter_routes(dataset1)
print("\nTask 1 Question 4 Result:")
print(result4)

# Task 1 Question 5
result5 = multiply_matrix(result1)
print("\nTask 1 Question 5 Result:")
print(result5)

# Task 1 Question 6
result6 = time_check(dataset2)
print("\nTask 1 Question 6 Result:")
print(result6)

