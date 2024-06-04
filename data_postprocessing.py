import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# important to know before you run:
# - change the file path to the data directory
# - change the file name to the data file name
# - change the self_consistency value according to the dataset you are running on
# - comment or uncomment the answer line in find_fallacy depending on the prompt type

# Used to perform regex on an input string
def contains_whole_word(large_string, word):
    pattern = rf'\b{re.escape(word)}\b'
    return bool(re.search(pattern, large_string))


# Equivalence dictionary that can be used to relabel model outputs that are paraphrased or use synonyms
fallacy_types = {
    'ad hominem': {'ad hominem'},
    'ad populum': {'ad populum', 'appeal to popularity', 'appeal to majority', 'bandwagon', 'appeal to popular belief',
                   'consensus', 'majority'},
    'appeal to (false) authority': {'appeal to (false) authority', 'appeal to false authority', 'appeal to authority',
                                    'misplaced authority', 'fallacy of authority', 'authority'},
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
    'false causality': {'false causality', 'false cause', 'correlation does not imply causation', 'causality',
                        'post hoc'},
    'false dilemma': {'false dilemma', 'black-and-white', 'black-or-white', 'black and white', 'black or white'},
    'guilt by association': {'guilt by association'},
    'hasty generalization': {'hasty generalization', 'generalization', 'generalizability'},
    'nothing': {'nothing', 'no fallacy', 'n/a'},
    'slippery slope': {'slippery slope'},
    'straw man': {'straw man', 'strawman', 'fallacy of extension'},
    'tu quoque': {'tu quoque', 'you too fallacy', 'appeal to hypocrisy', 'whataboutism'}

}


# Function that takes in an input text and outputs the fallacies detected as a list or 'no match'
def find_fallacy(input_sentence):
    # Normalize the input sentence to lowercase for case-insensitive matching
    input_sentence = input_sentence.lower()
    answer = ''

    # Searches for the entire sentence that follows after 'answer', used for CoT, CoT-SC and ToT
    # to avoid taking fallacy labels in the output into account that are not part of the final
    # conclusion of the model
    pattern = r'answer\s*(.*)'
    matches = []
    no_match = ['no match']
    # Perform the search
    match = re.search(pattern, input_sentence)
    if match:
        answer = match.group(1)

    # For RaR, uncomment the following line
    # answer = input_sentence #uncomment this if you want to run on the entire input

    # Check each fallacy and its synonyms
    for canonical, synonyms in fallacy_types.items():

        # Create a regex pattern that matches any of the synonyms in the equivalence dictionary
        pattern = r'\b(' + '|'.join(re.escape(syn) for syn in synonyms) + r')\b'

        # Search for the pattern in the input text (either the sentence after answer or the whole model output)
        if re.search(pattern, answer):

            # Only append new matches if the match has not yet been found (so there will be no duplicates)
            if canonical not in matches:
                # Create a list of all detected fallacies in the text
                matches.append(canonical)

    # Return the list of matches or 'no match'
    return matches if matches else no_match


# Change the path to the data directory and change the filename to the filename of the data (without extension)
path = '/Users/lukevandenwittenboer/PycharmProjects/LTP/test/results/results-mafalda/'
filename = 'results-cot-sc-openchat'
self_concistency = True  # Turn True if testing cot-cs, else turn False
results = pd.read_csv(path + filename + '.csv')

if self_concistency is True:

    # Iterate through all results
    for index, row in results.iterrows():

        labels = []
        # Find the labels in the three model answers for each statement for Self Consistency
        labels.extend(find_fallacy(row['actual_label1']))
        labels.extend(find_fallacy(row['actual_label2']))
        labels.extend(find_fallacy(row['actual_label3']))

        # Count the total amount of individual labels detected in each model output
        counts = Counter(labels)

        # If there are two or more labels that are identical, it
        matches = [label for label, count in counts.items() if count >= 2]

        # If no items appear twice or more, add the first label to matches
        if not matches and labels:  # Also checking if labels is not empty
            matches.append(labels[0])

        # Add the canonical labels to the results dataframe
        results.at[index, 'canonical'] = matches
else:
    # Go through all model outputs and detect the labels, add the canonical labels to the results dataframe
    results['canonical'] = results['actual_label'].apply(lambda x: find_fallacy(x))

# Split instances that have multiple matches into different rows in the dataframe
results = results.explode('canonical')

# Drop all instances that have n/a in the dataframe (usually not the case)
results = results.dropna(subset=['canonical'])

# Print the head of the dataframe (testing only)
print(results.head())

# Write the results including the canonical detected labels to a csv file
results.to_csv(path + filename + '_canonical.csv', index=False)

# Go through all sentences (testing only)
# for index, row in results.iterrows():
#     model_output = row['actual_label']
#
#     # Check which fallacy was detected
#     detected_label = find_fallacy(model_output)
#
#     # Print fallacies that were not detected (for analysis testing only)
#     # if detected_label is 'no match':
#     #     if row['expected_label'] != 'nothing':
#     #         print(answer + '   correct: ' + row['expected_label'])
#     processed_labels.append(detected_label)
#     if detected_label == row['expected_label']:
#         test_results.append(True)
#     else:
#         test_results.append(False)
#         print(detected_label + '   correct: ' + row['expected_label'])

# Calculate metrics
accuracy = accuracy_score(results['expected_label'], results['canonical'])
f1 = f1_score(results['expected_label'], results['canonical'], average='weighted')  # adjust average method as necessary

# Print the performance metrics
print(filename + '\t', accuracy, '\t', f1)
