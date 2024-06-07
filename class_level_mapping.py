import pandas as pd
import re
import numpy
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score




def level_1_mapping(column):
    level_1 = {
        'hasty generalization': 'fallacy of logic',
        'slippery slope': 'fallacy of logic',
        'causal oversimplification': 'fallacy of logic',
        'appeal to ridicule': 'appeal to emotion',
        'appeal to nature': 'fallacy of credibility',
        'false causality': 'fallacy of logic',
        'ad populum': 'fallacy of credibility',
        'ad hominem': 'fallacy of credibility',
        'false analogy': 'fallacy of logic',
        'false dilemma': 'fallacy of credibility',
        'appeal to fear': 'appeal to emotion',
        'appeal to authority': 'fallacy of credibility',
        'appeal to worse problem': 'appeal to emotion',
        'circular reasoning': 'fallacy of logic',
        'guilt by association': 'fallacy of credibility',
        'appeal to anger': 'appeal to emotion',
        'straw man': 'fallacy of logic',
        'appeal to tradition': 'fallacy of credibility',
        'equivocation': 'fallacy of logic',
        'fallacy of division': 'fallacy of logic',
        'tu quoque': 'fallacy of credibility',
        'appeal to positive emotion': 'appeal to emotion',
        'appeal to pity': 'appeal to emotion',
        'appeal to emotion (level 1)': 'appeal to emotion'

    }
    return column.map(level_1).fillna(column)


def level_0_mapping(column):
    level_0 = {
        'hasty generalization': 'fallacy detected',
        'slippery slope': 'fallacy detected',
        'causal oversimplification': 'fallacy detected',
        'appeal to ridicule': 'fallacy detected',
        'appeal to nature': 'fallacy detected',
        'false causality': 'fallacy detected',
        'ad populum': 'fallacy detected',
        'ad hominem': 'fallacy detected',
        'false analogy': 'fallacy detected',
        'false dilemma': 'fallacy detected',
        'appeal to fear': 'fallacy detected',
        'appeal to authority': 'fallacy detected',
        'appeal to worse problem': 'fallacy detected',
        'circular reasoning': 'fallacy detected',
        'guilt by association': 'fallacy detected',
        'appeal to anger': 'fallacy detected',
        'strawman': 'fallacy detected',
        'appeal to tradition': 'fallacy detected',
        'equivocation': 'fallacy detected',
        'fallacy of division': 'fallacy detected',
        'tu quoque': 'fallacy detected',
        'appeal to positive emotion': 'fallacy detected',
        'appeal to pity': 'fallacy detected',
        'appeal to emotion (level 1)': 'fallacy detected'
    }

    return column.map(level_0).fillna('no fallacy detected')

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

file_path = 'results-cot-sc-mistral_canonical.csv'  # Replace with the actual file path
results = pd.read_csv(file_path)


# Map actual_label column using the level_1 dictionary
results['expected_label_level1'] =  level_1_mapping(results['expected_label'])
results['expected_label_level2'] = level_0_mapping(results['expected_label'])

results['canonical_level1'] = level_1_mapping(results['canonical'])

print(results['canonical_level1'].head())
