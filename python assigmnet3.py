#!/usr/bin/env python
# coding: utf-8

# In[7]:


#EXCERSISE 1:PRIME NUMBER

number = int(input("Enter a number: "))

if number > 1:
    is_prime = True
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            is_prime = False
            break

    if is_prime:
        print(number, "is a prime number.")
    else:
        print(number, "is not a prime number.")
else:
    print(number, "is not a prime number.")


# In[9]:


#EXCERSISE2:-Product of random numbers

import random

# Generate two random numbers
num1 = random.randint(1, 10)
num2 = random.randint(1, 10)

# Calculate the product of the two numbers
product = num1 * num2

# Ask the user to enter the product of the numbers
user_answer = int(input(f"What is the product of {num1} and {num2}? "))

# Check if the user's answer is correct
if user_answer == product:
    print("Congratulations! Your answer is correct.")
else:
    print("Sorry, your answer is incorrect. The correct answer is:", product)


# In[10]:


#Exercise 3: Squares of Even/Odd Numbers

print("Squares of even numbers within the range of 100 to 200:")
for num in range(100, 201):
    if num % 2 == 0:  # Check if the number is even
        print(num ** 2)


# In[11]:


#excersise 4 :wordcounter
def count_words(text):
    # Split the text into words
    words = text.split()

    # Create a dictionary to store word counts
    word_counts = {}

    # Count the occurrence of each word
    for word in words:
        word = word.strip(",.!?")  # Remove punctuation marks
        word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts

def main():
    input_text = "This is a sample text. This text will be used to demonstrate the word counter."
    word_counts = count_words(input_text)

    # Print the word counts
    for word, count in word_counts.items():
        print(f"'{word}': {count}")

if __name__ == "__main__":
    main()


# In[13]:


#Excersise 5:Check For the Palindrome
# function which return reverse of a string

def isPalindrome(s):
	return s == s[::-1]


# Driver code
s = "malayalam"
ans = isPalindrome(s)

if ans:
	print("Yes")
else:
	print("No")


# In[ ]:




