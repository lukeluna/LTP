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
        'strawman': 'fallacy of logic',
        'appeal to tradition': 'fallacy of credibility',
        'equivocation': 'fallacy of logic',
        'fallacy of division': 'fallacy of logic',
        'tu quoque': 'fallacy of credibility',
        'appeal to positive emotion': 'appeal to emotion',
        'appeal to pity': 'appeal to emotion'

    }
    return column.map(level_1)


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
        'appeal to pity': 'fallacy detected'
    }

    return column.map(level_0).fillna('no fallancy detected')


file_path = 'results-rar_2-openchat.csv'  # Replace with the actual file path
results = pd.read_csv(file_path)


# Map actual_label column using the level_1 dictionary
results['expected_label_level1'] =  level_1_mapping(results['expected_label'])
results['expected_label_level2'] = level_0_mapping(results['expected_label'])

print(results.head())
