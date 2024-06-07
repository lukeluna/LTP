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
path = '/Users/simba/PycharmProjects/LanguageTechnology/'

# change this to the name of the txt file with your prompt (in the working directory)
filename = '''RaR_file1'''

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
nr_of_samples = 1

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

results = pd.DataFrame()
expected_labels = []
texts = []
actual_labels = []
start_time = time.time()

def rephrase_question(question):
    rephrase_prompt = f"Rephrase and expand the following question to improve clarity and detail: {question}"
    rephrase_response = ollama.chat(model='mistral:latest', messages=[
        {
            'role': 'user',
            'content': rephrase_prompt
        }
    ])
    rephrased_question = rephrase_response['message']['content'].strip()
    return rephrased_question

def get_response(rephrased_question):
    response_prompt = rephrased_question
    response = ollama.chat(model='mistral:latest', messages=[
        {
            'role': 'user',
            'content': response_prompt
        }
    ])
    return response['message']['content'].strip().lower()

with open('test_1.txt', 'w') as file:
    # Iterate over each text and its expected label
    examples = list_of_tuples
    counter = 1

    for text, expected_label in examples:
        print(f"test {counter} of {len(list_of_tuples)}")
        print(f'Text: {text}\n')
        print(f'Expected: {expected_label}\n')

        message_content = message_base_content.format(text)

        # Rephrase the question
        rephrased_question = rephrase_question(message_content)

        print(rephrased_question)

        # Get the response using the rephrased question
        actual_label = get_response(rephrased_question)

        print(actual_label)
        test_passed = contains_whole_word(actual_label, 'answer: ' + expected_label)

        # Log expected and actual labels and text
        expected_labels.append(expected_label)
        actual_labels.append(actual_label)
        texts.append(text)

        # Append test result
        results = results._append({
            'text': text,
            'expected_label': expected_label,
            'actual_label': actual_label,
            'result': test_passed
        }, ignore_index=True)

        counter += 1

    # Save results to file
    results.to_csv('results.csv', index=False)


