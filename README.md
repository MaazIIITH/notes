EXPERIMENT-2
	
Aim of the experiment:
	


Theory: 








Source Code: 



squares = []
for i in range(1, 31):
squared = i ** 2
squares.append(squared)
print(squares[5:])


Output: 




						EXPERIMENT-3a


Source Code: 

def reverse_string_for_loop(string):
    reversed_string = ""
    for i in range(len(string) - 1, -1, -1):
        reversed_string += string[i]
    return reversed_string

def reverse_string_while_loop(string):
    reversed_string = ""
    index = len(string) - 1
    while index >= 0:
        reversed_string += string[index]
        index -= 1
    return reversed_string

# Example usage: 1 
original_string = "Hello_there"
reversed_string_for = reverse_string_for_loop(original_string)
reversed_string_while = reverse_string_while_loop(original_string)

print("Original String:", original_string)
print("Reversed String (For Loop):", reversed_string_for)
print("Reversed String (While Loop):", reversed_string_while)

Output: 





						

    
 EXPERIMENT-3b


Source Code: 

# Get input number from the user
num = int(input("Enter a number: "))

# Initialize a variable to store the sum of digits
sum_of_digits = 0

# While the number is greater than 0, repeat the following:
while num > 0:
    # Extract the last digit of the number
    digit = num % 10

    # Add the extracted digit to the sum
    sum_of_digits += digit

    # Remove the last digit from the number
    num = num // 10

# Print the calculated sum of digits
print("Sum of digits:", sum_of_digits)
 
					       EXPERIMENT-3c


Source Code: 

# Get input number from the user
num = int(input("Enter a number: "))

# Initialize a variable to store the factorial
factorial = 1

# Calculate the factorial using a for loop
for i in range(1, num + 1):
    factorial *= i

# Print the calculated factorial
print("Factorial of", num, "is:", factorial)

Output: 
 
     					            EXPERIMENT-3d


Source Code: 

# Get the number of terms from the user
num = int(input("Enter Number of terms: "))

# Initialize the first two Fibonacci numbers
num1 = 0
num2 = 1

# Print the first two Fibonacci numbers
print("Fibonacci Series: ", num1, num2, end=" ")

# Generate and print the remaining Fibonacci numbers
for i in range(2, num + 1):
    # Calculate the next Fibonacci number
    num3 = num1 + num2

    # Print the next Fibonacci number
    print(num3, end=" ")

    # Update the first two numbers for the next iteration
    num1 = num2
    num2 = num3

Output: 
 
           					      EXPERIMENT-3e



Source Code: 
# Get the number of rows from the user
n = int(input("Enter the number of rows: "))

# Print the pattern
for i in range(1, n + 1):
    # Print leading spaces
    print(" " * (n - i), end="")

    # Print asterisks
    print("* " * i)

Output: 
















Exp_4

Code
def max_of_three():
  a = int(input("Enter the first number: "))
  b = int(input("Enter the second number: "))
  c = int(input("Enter the third number: "))

  if a >= b and a >= c: 1 
    return a
  elif b >= a and b >= c:
    return b
  else:
    return c

# Example usage:
result = max_of_three()
print("The maximum number is:", result)
 
exp_5
Source Code: 

import random

# Define lists of story elements
characters = ["Caesar", "Roman", "Strawman"]
_11_plots = ["Once there used to be a king", "There was a strong man", "He used to rule the seas"]
_12_plots = ["Feared by all", "He was a great warrior", "He was a great leader"]
part_story = ["He wanted to rule the land", "He had an ambition of great wealth in his mind", "He had a vision of great future"]
final_part = ["So he pursued the plans he had set in his mind", "He was willing to do anything to get he what he sought", "So he aligned his forces to capture the land"]

# Generate the story
story = (
    f"\nSo the story goes...\n\n"
    f"+ {random.choice(_11_plots)} +\n"
    f'"{random.choice(characters)}" +\n'
    f'"{random.choice(part_story)}" +\n'
    f"{random.choice(final_part)}\n"
)

# Print the story
print(story)
 

Output: 









 
Exp-6


Source Code:
import pandas as pd
from faker import Faker
import random as rand

# Create a Faker object and set the seed
Fake = Faker()
Faker.seed(0)

# Set the random seed
rand.seed(0)

# Number of records to generate
num_records = 100

# Initialize lists to store the data
roll_no = []
name = []
age = []
gender = []
mpython = []
mjava = []

# Generate data
for i in range(num_records):
    roll_no.append(Fake.unique.random_int(0, 100))
    name.append(Fake.unique.name())
    age.append(Fake.random_int(18, 22))
    gender.append(Fake.random.choice(["Male", "Female"]))
    mpython.append(Fake.random_int(0, 100))
    mjava.append(Fake.random_int(0, 100))

# Create a DataFrame from the data
data = pd.DataFrame({
    "Roll No": roll_no,
    "Name": name,
    "Age": age,
    "Gender": gender,
    "Marks Python": mpython,
    "Marks Java": mjava
})

# Print the DataFrame
print(data)

# Save the DataFrame to a CSV file
data.to_csv("Synthetic_Data.csv") 		
 	                        
Output: 







   EXPERIMENT - 7

Aim of the experiment:


Theory: 	
A>
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("Random 2D array is:")
print(random_array)

B>
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("2D Array:")
print(random_array)

total_sum = np.sum(random_array)
print(f"Sum of all elements: {total_sum}")

C>
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("2D Array:")
print(random_array)

total_sum = np.sum(random_array)
print(f"Sum of all elements: {total_sum}")

max_value = np.max(random_array)
print(f"Maximum value in the array: {max_value}")

mean_value = np.mean(random_array)
print(f"Mean value of the array: {mean_value}")

D>
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("2D Array:")
print(random_array)

mean_value = np.mean(random_array)
print(f"Mean of array elements: {mean_value}")

E>
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("2D Array:")
print(random_array)

row_sums = np.sum(random_array, axis=1)
print("Sum of elements in each row:")
print(row_sums)

F>
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("Original 2D Array:")
print(random_array)

transposed_array = random_array.T

print("Transposed 2D Array:")
print(transposed_array)

G>
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("2D Array:")
print(random_array)

mask_greater_than_25 = random_array > 25
print("Boolean mask of elements > 25:")
print(mask_greater_than_25)

filtered_elements = random_array[mask_greater_than_25]
print("Elements greater than 25:")
print(filtered_elements)












EXPERIMENT-8

Aim of the experiment: Perform Statistics and Data Visualization in python. Assume you have a .csv file containing 10 student details along with their marks in python, java and C language. Perform following operations on it.
•	Print mean, standard deviation, minimum marks, maximum marks 1st quantile, 3rd quantile, maximum marks in each category.
•	Plot a histogram plot for each subject.

Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from faker import Faker

# Generate fake data
fake = Faker()
np.random.seed(42)

python_marks = np.random.randint(60, 101, size=10)
names = [fake.name() for _ in range(10)]
c_marks = np.random.randint(60, 101, size=10)
java_marks = np.random.randint(60, 101, size=10)

# Create a DataFrame
data = pd.DataFrame({
    'Name': names,
    'Python': python_marks,
    'C': c_marks,
    'Java': java_marks
})

# Save the DataFrame to a CSV file
data.to_csv('student_marks.csv', index=False)

# Read the CSV file
data = pd.read_csv('student_marks.csv')

# Calculate statistics
statistics = {}
subjects = ['Python', 'Java', 'C']

for subject in subjects:
    stats = {
        'Mean': data[subject].mean(),
        'Standard Deviation': data[subject].std(),
        'Minimum Marks': data[subject].min(),
        '1st Quantile': data[subject].quantile(0.25),
        'Maximum Marks': data[subject].max(),
        '3rd Quantile': data[subject].quantile(0.75)
    }
    statistics[subject] = stats

for subject, stats in statistics.items():
    print(f"Statistics for {subject}:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    print()

# Create histograms
for subject in subjects:
    plt.figure(figsize=(8, 5))
    plt.title(f'Histogram of {subject} Marks')
    plt.hist(data[subject], bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Marks')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()















 
EXPERIMENT NO – 9


Aim of the experiment: Design a Python program to implement Linear Regression House price prediction using california_housing from scikit-learn.

Theory:



Source Code
Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load California Housing Dataset
housing_data = fetch_california_housing()

# Create DataFrames for features (X) and target (y)
X = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
y = pd.Series(housing_data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize the actual vs. predicted house prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid()
plt.show()



































EXPERIMENT NO – 10

Aim of the experiment: Design a Python program to create a recommender system.

Theory:
Source Code:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'movie_id': [101, 102, 103, 101, 102, 103, 101, 102, 103],
    'rating': [5, 4, 3, 4, 5, 3, 3, 4, 5]
}

df = pd.DataFrame(data)

user_item_matrix = pd.pivot_table(df, values='rating', index='user_id', columns='movie_id')

train_matrix, test_matrix = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

nmf_model = NMF(n_components=2, random_state=42)
nmf_model.fit(train_matrix)

def make_recommendations(user_id, num_recommendations):
    user_factors = nmf_model.transform(train_matrix)[user_id - 1].reshape(1, -1)
    movie_factors = nmf_model.components_.T
    movie_similarities = cosine_similarity(user_factors, movie_factors).flatten()
    recommended_movies = np.argsort(-movie_similarities)[:num_recommendations]
    return recommended_movies

user_id = 1
num_recommendations = 3

recommended_movies = make_recommendations(user_id, num_recommendations)
print(f"Recommended movies for user {user_id}: {recommended_movies}")




		

EXPERIMENT NO – 11

Aim of the experiment: Write a program in Python to read a text file and write a text file.

Theory:
 
Code
# Read from a file
with open('input.txt', 'r') as file:
    data = file.read()
    print("Content of input.txt:", data)

# Write to a file
with open('output.txt', 'w') as file:
    file.write("This is a sample output.\n")
    file.write("Data read from input.txt:\n")
    file.write(data)










EXPERIMENT NO – 12

Aim of the experiment: Write a program in Python to implement exception handling.

Theory:

 
Code
def divide_numbers():
    try:
        numerator = float(input("Enter the numerator: "))
        denominator = float(input("Enter the denominator: "))
        result = numerator 1  / denominator
        print(f"The result of division is: {result}")
    except ZeroDivisionError:
        print("Error: You cannot divide by zero.")
    except ValueError:
        print("Error: Invalid input. Please enter numbers only.")
    finally:
        print("Execution complete.")

divide_numbers()
Experiment_13
Aim of the experiment: Data Science Project: students can take any dataset of their choice (titanic / stock price prediction / credit card fraud detection, etc.) and show all the steps of the data science life cycle.

Theory:
 
Code
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load Titanic dataset
data = sns.load_dataset("titanic")

# Data preprocessing
data['age'] = data['age'].fillna(data['age'].median())
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])

data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['embarked'] = data['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data['family_size'] = data['sibsp'] + data['parch'] + 1

# Exploratory Data Analysis (EDA)
# ... (Add your desired visualizations here)

# Feature selection and target variable
features = ['sex', 'age', 'pclass', 'fare', 'family_size', 'embarked']
target = 'survived'

# Split data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Example of making predictions on new data
new_data = pd.DataFrame({
    'sex': [0, 1],
    'age': [25, 30],
    'pclass': [1, 2],
    'fare': [100, 50],
    'family_size': [2, 3],
    'embarked': [0, 1]
})

new_predictions = model.predict(new_data)
print(new_predictions)

