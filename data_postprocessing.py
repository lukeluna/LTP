import re
import numpy
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def contains_whole_word(large_string, word):
    pattern = rf'\b{re.escape(word)}\b'
    return bool(re.search(pattern, large_string))

fallacy_types = {
    'ad hominem': {'ad hominem'},
    'ad populum': {'ad populum', 'appeal to popularity', 'appeal to majority', 'bandwagon', 'appeal to popular belief', 'consensus', 'majority'},
    'appeal to (false) authority': {'appeal to (false) authority', 'appeal to false authority', 'appeal to authority', 'misplaced authority', 'fallacy of authority', 'authority'},
    'appeal to anger': {'appeal to anger', 'appeal to emotion (anger)', 'anger appeal', 'anger'},
    'appeal to fear': {'appeal to fear', 'appeal to emotion (fear)', 'fear appeal', 'fear'},
    'appeal to nature': {'appeal to nature'},
    'appeal to emotion (level 1)': {'appeal to emotion', 'appeal to negative emotion'},
    'appeal to pity': {'appeal to pity', 'appeal to emotion (pity)'},
    'appeal to positive emotion': {'appeal to positive emotion', 'appeal to (positive) emotion'},
    'appeal to ridicule': {'appeal to ridicule'},
    'appeal to tradition': {'appeal to tradition', 'ad tradition'},
    'appeal to worse problems': {'appeal to worse problems', 'appeal to worse problem'},
    'causal oversimplification': {'causal oversimplification', 'cherry picking', 'oversimplification'},
    'circular reasoning': {'circular reasoning', 'begging the question', 'circular claim'},
    'equivocation': {'equivocation', 'vagueness'},
    'fallacy of division': {'fallacy of division'},
    'false analogy': {'false analogy', 'analogy'},
    'false causality': {'false causality', 'false cause', 'correlation does not imply causation', 'causality', 'post hoc'},
    'false dilemma': {'false dilemma', 'black-and-white', 'black-or-white', 'black and white', 'black or white'},
    'guilt by association': {'guilt by association'},
    'hasty generalization': {'hasty generalization', 'generalization', 'generalizability'},
    'nothing': {'nothing', 'no fallacy', 'n/a'},
    'slippery slope': {'slippery slope'},
    'straw man': {'straw man', 'strawman', 'fallacy of extension'},
    'tu quoque': {'tu quoque', 'you too fallacy', 'appeal to hypocrisy', 'whataboutism'}

}

def find_fallacy(input_sentence):
    # Normalize the input sentence to lowercase for case-insensitive matching
    input_sentence = input_sentence.lower()
    answer = ''
    pattern = r'answer:\s*(.*)' #TODO: this now requires the format answer: ... so it does not work for RaR or CoT

    # Perform the search
    match = re.search(pattern, input_sentence)
    if match:
        answer = match.group(1)

    #answer = input_sentence #uncomment this if you want to run on the entire input

    # Check each fallacy and its synonyms
    for canonical, synonyms in fallacy_types.items():
        # Create a regex pattern that matches any of the synonyms
        pattern = r'\b(' + '|'.join(re.escape(syn) for syn in synonyms) + r')\b'

        # Search for the pattern in the input sentence
        if re.search(pattern, answer):
            return canonical, answer  #TODO: option to find multiple classes in answer

    # Return None if no fallacy type is found
    return 'no match', answer

path = '/Users/lukevandenwittenboer/PycharmProjects/LTP/test/'

results = pd.read_csv(path + 'results/results-mafalda/results-tot-gemma.csv')

actual_labels = results['actual_label']
expected_labels = results['expected_label']

processed_labels = []
test_results = []

# Go through all sentences
for index, row in results.iterrows():
    model_output = row['actual_label']

    # Check which fallacy was detected
    detected_label, answer = find_fallacy(model_output)

    # Print fallacies that were not detected (for analysis testing only)
    # if detected_label is 'no match':
    #     if row['expected_label'] != 'nothing':
    #         print(answer + '   correct: ' + row['expected_label'])
    processed_labels.append(detected_label)
    if detected_label == row['expected_label']:
        test_results.append(True)
    else:
        test_results.append(False)
        print(detected_label + '   correct: ' + row['expected_label'])

#
# Calculate metrics
accuracy = accuracy_score(expected_labels, processed_labels)
precision = precision_score(expected_labels, processed_labels, average='weighted')  # adjust average method as necessary
recall = recall_score(expected_labels, processed_labels, average='weighted')  # adjust average method as necessary
f1 = f1_score(expected_labels, processed_labels, average='weighted')  # adjust average method as necessary
#

print(accuracy, f1, precision, recall)

# #
# accuracy = sum(test_results) / len(test_results)
# f1 = f1_score([True] * len(test_results), test_results, average='weighted')
# precision = precision_score([True] * len(test_results), test_results, average='weighted')
# recall = recall_score([True] * len(test_results), test_results, average='weighted')
#
# print(accuracy, f1, precision, recall)


