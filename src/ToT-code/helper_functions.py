import random
from collections import defaultdict

def sample_uniform_rows(data, num_samples):
    # Group data by labels
    label_groups = defaultdict(list)
    for item in data:
        label_groups[item[1]].append(item)
    
    # Sample uniformly
    sampled_data = []
    for label, items in label_groups.items():
        if len(items) > num_samples:
            sampled_items = random.sample(items, num_samples)
        else:
            sampled_items = items  # If not enough items, take all
        sampled_data.extend(sampled_items)
    
    return sampled_data
