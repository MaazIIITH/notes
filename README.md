# Experiments

## EXPERIMENT-2

### Aim of the experiment:



### Theory:





### Source Code:

```python
squares = []
for i in range(1, 31):
    squared = i ** 2
    squares.append(squared)
print(squares[5:])
```

### Output:



---

## EXPERIMENT-3a

### Source Code:

```python
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

# Example usage:
original_string = "Hello_there"
reversed_string_for = reverse_string_for_loop(original_string)
reversed_string_while = reverse_string_while_loop(original_string)

print("Original String:", original_string)
print("Reversed String (For Loop):", reversed_string_for)
print("Reversed String (While Loop):", reversed_string_while)
```

### Output:



---

## EXPERIMENT-3b

### Source Code:

```python
# Get input number from the user
num = int(input("Enter a number: "))

# Initialize a variable to store the sum of digits
sum_of_digits = 0

# While the number is greater than 0, repeat the following:
while num > 0:
    digit = num % 10
    sum_of_digits += digit
    num = num // 10

# Print the calculated sum of digits
print("Sum of digits:", sum_of_digits)
```

---

## EXPERIMENT-3c

### Source Code:

```python
# Get input number from the user
num = int(input("Enter a number: "))

# Initialize a variable to store the factorial
factorial = 1

# Calculate the factorial using a for loop
for i in range(1, num + 1):
    factorial *= i

# Print the calculated factorial
print("Factorial of", num, "is:", factorial)
```

---

## EXPERIMENT-3d

### Source Code:

```python
# Get the number of terms from the user
num = int(input("Enter Number of terms: "))

# Initialize the first two Fibonacci numbers
num1 = 0
num2 = 1

# Print the first two Fibonacci numbers
print("Fibonacci Series: ", num1, num2, end=" ")

# Generate and print the remaining Fibonacci numbers
for i in range(2, num + 1):
    num3 = num1 + num2
    print(num3, end=" ")
    num1 = num2
    num2 = num3
```

---

## EXPERIMENT-3e

### Source Code:

```python
# Get the number of rows from the user
n = int(input("Enter the number of rows: "))

# Print the pattern
for i in range(1, n + 1):
    print(" " * (n - i), end="")
    print("* " * i)
```

---

## EXPERIMENT-4

### Source Code:

```python
def max_of_three():
    a = int(input("Enter the first number: "))
    b = int(input("Enter the second number: "))
    c = int(input("Enter the third number: "))

    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c

# Example usage:
result = max_of_three()
print("The maximum number is:", result)
```

---

## EXPERIMENT-5

### Source Code:

```python
import random

# Define lists of story elements
characters = ["Caesar", "Roman", "Strawman"]
plots_11 = ["Once there used to be a king", "There was a strong man", "He used to rule the seas"]
plots_12 = ["Feared by all", "He was a great warrior", "He was a great leader"]
part_story = ["He wanted to rule the land", "He had an ambition of great wealth in his mind", "He had a vision of great future"]
final_part = ["So he pursued the plans he had set in his mind", "He was willing to do anything to get what he sought", "So he aligned his forces to capture the land"]

# Generate the story
story = (
    f"\nSo the story goes...\n\n"
    f"{random.choice(plots_11)}\n"
    f"{random.choice(characters)}\n"
    f"{random.choice(part_story)}\n"
    f"{random.choice(final_part)}\n"
)

# Print the story
print(story)
```

---

## EXPERIMENT-6

### Source Code:

```python
import pandas as pd
from faker import Faker
import random as rand

Fake = Faker()
Faker.seed(0)
rand.seed(0)

num_records = 100
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

data = pd.DataFrame({
    "Roll No": roll_no,
    "Name": name,
    "Age": age,
    "Gender": gender,
    "Marks Python": mpython,
    "Marks Java": mjava
})

print(data)
data.to_csv("Synthetic_Data.csv")
```

---

Here's how you can format **Experiment 7** for a report or README:

---

### **EXPERIMENT - 7**

**Aim of the Experiment:**  
To explore and perform various operations on 2D arrays using the NumPy library in Python.  

---

**Theory:**  
NumPy is a powerful Python library used for numerical computations. It provides support for creating and manipulating arrays, performing statistical operations, and working with multidimensional data structures. This experiment demonstrates several key operations on 2D arrays, such as generating arrays, computing sums, means, and transposing arrays.

---

### **Code Snippets and Descriptions**

#### **A> Generate a Random 2D Array**
```python
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("Random 2D array is:")
print(random_array)
```
**Description:**  
- Creates a 4x5 2D array of random integers between 1 and 50.

---

#### **B> Calculate the Total Sum of Array Elements**
```python
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("2D Array:")
print(random_array)

total_sum = np.sum(random_array)
print(f"Sum of all elements: {total_sum}")
```
**Description:**  
- Computes and displays the total sum of all elements in the array.

---

#### **C> Calculate Sum, Maximum, and Mean of Array Elements**
```python
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
```
**Description:**  
- Computes the sum, maximum value, and mean value of the array elements.

---

#### **D> Calculate the Mean of Array Elements**
```python
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("2D Array:")
print(random_array)

mean_value = np.mean(random_array)
print(f"Mean of array elements: {mean_value}")
```
**Description:**  
- Calculates and displays the mean of the array elements.

---

#### **E> Compute Row-Wise Sums**
```python
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("2D Array:")
print(random_array)

row_sums = np.sum(random_array, axis=1)
print("Sum of elements in each row:")
print(row_sums)
```
**Description:**  
- Calculates the sum of array elements for each row separately.

---

#### **F> Transpose a 2D Array**
```python
import numpy as np

random_array = np.random.randint(1, 51, size=(4, 5))

print("Original 2D Array:")
print(random_array)

transposed_array = random_array.T

print("Transposed 2D Array:")
print(transposed_array)
```
**Description:**  
- Transposes the 2D array (i.e., swaps rows and columns).

---

#### **G> Filter Array Elements Greater Than 25**
```python
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
```
**Description:**  
- Filters and displays elements of the array greater than 25 using a boolean mask.

---

**Conclusion:**  
This experiment highlights the power and versatility of NumPy for array manipulation and statistical operations. Each code snippet demonstrates a unique operation, reinforcing the concepts of data handling in Python.

--- 

You can copy this for your README or documentation!


Here’s how you can format **Experiment 8** for a report or README:

---

### **EXPERIMENT - 8**

**Aim of the Experiment:**  
Perform statistical analysis and data visualization in Python using a `.csv` file containing details of 10 students and their marks in Python, Java, and C programming languages.

---

### **Tasks Performed**  

1. **Statistical Analysis**:  
   - Calculate mean, standard deviation, minimum marks, maximum marks, 1st quantile, and 3rd quantile for each subject.
   - Identify maximum marks in each category.
   
2. **Data Visualization**:  
   - Generate histogram plots for the marks distribution in each subject.

---

### **Code Implementation**

```python
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
```

---

### **Steps Explained**

1. **Generating Fake Data**:  
   - Used the `Faker` library to create random names.  
   - Generated random marks for Python, Java, and C programming between 60 and 100.  
   - Saved the data into a `.csv` file named `student_marks.csv`.  

2. **Statistical Analysis**:  
   - Calculated mean, standard deviation, minimum, maximum, 1st quantile (25th percentile), and 3rd quantile (75th percentile) for each subject using pandas.  
   - Stored the results in a dictionary for easy printing.

3. **Data Visualization**:  
   - Created histogram plots for each subject's marks to visualize the distribution of marks.  
   - Configured the plots with titles, labeled axes, and gridlines for better readability.

---

### **Sample Output**  

#### **Statistical Analysis Output**
```
Statistics for Python:
Mean: 80.2
Standard Deviation: 12.2
Minimum Marks: 60
1st Quantile: 72.0
Maximum Marks: 100
3rd Quantile: 90.5

Statistics for Java:
Mean: 79.6
Standard Deviation: 10.8
Minimum Marks: 63
1st Quantile: 72.5
Maximum Marks: 95
3rd Quantile: 89.75

Statistics for C:
Mean: 81.3
Standard Deviation: 11.1
Minimum Marks: 64
1st Quantile: 74.25
Maximum Marks: 97
3rd Quantile: 92.0
```

#### **Histogram Example (Python Marks)**

- A histogram visualizing the frequency of marks in Python, split into bins.

---

### **Conclusion**  
This experiment demonstrates the application of Python libraries like Pandas, NumPy, and Matplotlib for performing statistical analysis and visualizing data. These tools are essential for understanding datasets and uncovering insights through computations and plots. 

--- 

You can copy-paste this into your report!

## EXPERIMENT-9

### Aim of the experiment:
Design a Python program to implement Linear Regression House Price prediction using California housing data from scikit-learn.

### Source Code:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Linear Regression code with California housing dataset
```

---

## EXPERIMENT-10

### Aim of the experiment:
Design a Python program to create a recommender system...

(Continue with the provided experiment details)
