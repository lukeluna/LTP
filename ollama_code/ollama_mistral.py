import time
import ollama
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Define the base message content
message_base_content = '''
Definition:
An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument. 
A fallacious argument is an argument where the premises do not entail the conclusion. 

Types of fallacy:
hasty generalization 
slippery slope
causal oversimplification
appeal to ridicule
appeal to nature
false causality
ad populum
ad hominem
false analogy
false dilemma
appeal to fear
appeal to (false) authority
appeal to worse problems
circular reasoning
guilt by association
appeal to anger                 
straw man
appeal to tradition 
equivocation 
fallacy of division 
tu quoque 
appeal to positive emotion 
appeal to pity

I will give you different texts. For each text, determine if it is a fallacy or not. If it is not, or cannot be judged based on the text, write 'nothing'. If it is a fallacy, tell me which fallacy type of the above list it is. Do not give any explanation, just write the answer. If you are unsure, write the one answer that seems most likely: 
{}
'''

# List of example texts and their expected labels
examples = [
    ('Why is the sky blue?', 'nothing'),
    ('You have no idea how street is, you better stop talking!', 'ad hominem'),
    ("You're probably a drug addict yourself.", 'ad hominem'),
    ("Why do you want a university-doctor's title to be shown on their identification-card? You must be one yourself, you egotistic prick ! Does it make you feel respected that we all know your little title?", "ad hominem"),
    ("You’re way too nervous about driving at night so of course you don’t want to drive.", "ad hominem"),
    ("Three million fans can't be wrong!", 'ad populum'),
    ("The human soul is immortal, because all learned men agree that anything which does not come out of the potentiality of matter is incorruptible and immortal.", "ad populum"),
    ("We all believe such preachers as Mr. Raskin. He is so nearly right, his ideals are so very high, that most people assent — while they have no difficulty in evading them and going on their way as if a breath of wind had fanned their faces, and no voice of truth had stirred their spirits.", "ad populum"),
    ("You should buy this phone; it's the best-selling model worldwide.","ad populum"),
    ("Are you tired of being ignored by your government? Is it right that the top 1% have so much when the rest of us have so little? I urge you to vote for me today!", 'appeal to anger'),
    ("I get mad when i think about all these poor guys on the street, having no home, no job, no family. All because they started taking drugs. Its so sad, we need heavier penalties.","appeal to anger"),
]
test_results = []
start_time = time.time() 
# Iterate over each text and its expected label
for text, expected_label in examples:
    message_content = message_base_content.format(text)
    
     # Start the timer
    response = ollama.chat(model='mistral:latest', messages=[
        {
            'role': 'user',
            'content': message_content
        }
    ])
    end_time = time.time()  # End the timer
    
    actual_label = response['message']['content'].strip()
    actual_label = actual_label.lower()
    test_passed = contains_whole_word(actual_label, expected_label)
    test_results.append(test_passed)
    
    # Print the response and test result
    print(f'Text: {text}')
    print(f'Expected: {expected_label}')
    print(f'Actual: {actual_label}')
    print(f'Test Passed:', test_passed)
    print('----------------------------')
    
accuracy = sum(test_results) / len(test_results)
f1 = f1_score([True]*len(test_results), test_results, average='weighted')
precision = precision_score([True]*len(test_results), test_results, average='weighted')
recall = recall_score([True]*len(test_results), test_results, average='weighted')




print(f'Time Taken: {end_time - start_time:.2f} seconds\n')
