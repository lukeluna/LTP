import ollama
import time
import re
import numpy
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def contains_whole_word(large_string, word):
    pattern = rf'\b{re.escape(word)}\b'
    return bool(re.search(pattern, large_string))

# change this to your working directory
path = '/Users/lukevandenwittenboer/PycharmProjects/LTP/test/'

# change this to the name of the txt file with your prompt (in the working directory)
filename = '''test'''
filename = '''zero_shot_tot_2'''
with open(path + filename + '.txt', 'r') as file:
    message_base_content = file.read()
file.close()

# read all class files
ad_hominem = pd.read_csv(path + 'data/handmade/ad_hominem_final.csv')
ad_populum = pd.read_csv(path + 'data/handmade/ad_populum_final.csv')
appeal_to_anger = pd.read_csv(path + 'data/handmade/appeal_to_anger_final.csv')
appeal_to_authority = pd.read_csv(path + 'data/handmade/appeal_to_authority_final.csv')
appeal_to_fear = pd.read_csv(path + 'data/handmade/appeal_to_fear_final.csv')
appeal_to_nature = pd.read_csv(path + 'data/handmade/appeal_to_nature_final.csv')
appeal_to_pity = pd.read_csv(path + 'data/handmade/appeal_to_pity_final.csv')
appeal_to_positive_emotion = pd.read_csv(path + 'data/handmade/appeal_to_positive_emotion_final.csv')
appeal_to_ridicule = pd.read_csv(path + 'data/handmade/appeal_to_ridicule_final.csv')
appeal_to_tradition = pd.read_csv(path + 'data/handmade/appeal_to_tradition_final.csv')
appeal_to_worse_problems = pd.read_csv(path + 'data/handmade/appeal_to_worse_problems_final.csv')
causal_oversimplifiation = pd.read_csv(path + 'data/handmade/causal_oversimplification_final.csv')
#circular_reasoning =
equivocation = pd.read_csv(path + 'data/handmade/equivocation_final.csv')
fallacy_of_division = pd.read_csv(path + 'data/handmade/fallacy_of_division_final.csv')
false_analogy = pd.read_csv(path + 'data/handmade/false_analogy_final.csv')
false_causality = pd.read_csv(path + 'data/handmade/false_causality_final.csv')
false_dilemma = pd.read_csv(path + 'data/handmade/false_dilemma_final.csv')
#guilt_by_association =
hasty_generalization = pd.read_csv(path + 'data/handmade/hasty_generalization_final.csv')
nothing = pd.read_csv(path + 'data/handmade/nothing_final.csv')
slippery_slope = pd.read_csv(path + 'data/handmade/slippery_slope_final.csv')
strawman = pd.read_csv(path + 'data/handmade/strawman_final.csv')
#tu_quoque =

list_of_tuples = []

datasets = [ad_hominem, ad_populum, appeal_to_anger, appeal_to_authority, appeal_to_fear, appeal_to_nature,
            appeal_to_pity, appeal_to_ridicule, appeal_to_tradition, appeal_to_worse_problems, causal_oversimplifiation,
#            circular_reasoning, guilt_by_association, tu_quoque,
            equivocation, fallacy_of_division, false_analogy, false_causality, false_dilemma,
            hasty_generalization, nothing, slippery_slope, strawman]

# define the number of samples per class you want to use.
nr_of_samples = 3

for dataset in datasets:
    if len(dataset) >= nr_of_samples:
        # Sample 15 entries if the dataset is large enough
        sampled = dataset.sample(n=nr_of_samples, random_state=random.randint(1, 100))  # Change seed for true randomness
    else:
        # If the dataset has less than 15 entries, take all available data
        sampled = dataset

    # Convert the sampled data to tuples and add to the list
    list_of_tuples.extend(sampled.itertuples(index=False, name=None))

# Print the first few tuples
print(list_of_tuples[:5])

test_results = []
start_time = time.time()
# Iterate over each text and its expected label
examples = list_of_tuples
counter = 1
for text, expected_label in examples:
    print(counter, len(list_of_tuples))
    message_content = message_base_content.format(text)

    # Start the timer
    response = ollama.chat(model='mistral:latest', messages=[
        {
            'role': 'user',
            'content': message_content
        }
    ])
    end_time = time.time()  # End the timer
    counter = counter + 1
    actual_label = response['message']['content'].strip()
    actual_label = actual_label.lower()
    test_passed = contains_whole_word(actual_label, 'answer: ' + expected_label)
    test_results.append(test_passed)

    # Print the response and test result
    print(f'Text: {text}')
    print(f'Expected: {expected_label}')
    print(f'Actual: {actual_label}')
    print(f'Test Passed:', test_passed)
    print('----------------------------')

accuracy = sum(test_results) / len(test_results)
f1 = f1_score([True] * len(test_results), test_results, average='weighted')
precision = precision_score([True] * len(test_results), test_results, average='weighted')
recall = recall_score([True] * len(test_results), test_results, average='weighted')

print(f'Time Taken: {end_time - start_time:.2f} seconds\n')

print(accuracy, f1, precision, recall)