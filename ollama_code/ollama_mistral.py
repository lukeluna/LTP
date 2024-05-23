import time
import ollama
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re

#creating the dataset

ad_hominem = pd.read_csv('data/handmade data/ad_hominem_final.csv')
ad_populum = pd.read_csv('data/handmade data/ad_populum_final.csv')
appeal_to_anger = pd.read_csv('data/handmade data/appeal_to_anger_final.csv')
appeal_to_authority = pd.read_csv('data/handmade data/appeal_to_authority_final.csv')
#appeal_to_fear = pd.read_csv('data/handmade data/appeal_to_fear_final.csv')
appeal_to_nature = pd.read_csv('data/handmade data/appeal_to_nature_final.csv')
appeal_to_pity = pd.read_csv('data/handmade data/appeal_to_pity_final.csv')
#appeal_to_ridicule = pd.read_csv('data/handmade data/appeal_to_ridicule_final.csv')
appeal_to_tradition = pd.read_csv('data/handmade data/appeal_to_tradition_final.csv')
appeal_to_worse_problems = pd.read_csv('data/handmade data/appeal_to_worse_problems_final.csv')
causal_oversimplifiation = pd.read_csv('data/handmade data/causal_oversimplification_final.csv')
equivocation = pd.read_csv('data/handmade data/equivocation_final.csv')
fallacy_of_division = pd.read_csv('data/handmade data/fallacy_of_division_final.csv')
false_analogy = pd.read_csv('data/handmade data/false_analogy_final.csv')
false_causality = pd.read_csv('data/handmade data/false_causality_final.csv')
false_dilemma = pd.read_csv('data/handmade data/false_dilemma_final.csv')
hasty_generalization = pd.read_csv('data/handmade data/hasty_generalization_final.csv')
nothing = pd.read_csv('data/handmade data/nothing_final.csv')
slippery_slope = pd.read_csv('data/handmade data/slippery_slope_final.csv')
strawman = pd.read_csv('data/handmade data/strawman_final.csv')

data_tuples_fear = [
    ("Nuclear power was the reason of death of millions of people. It should vanish from this planet", "appeal to fear"),
    ("Yes, all the polar-bears are dying, and we are next.", "appeal to fear"),
    ("Yes,,you certainly don't want your children to die from this virus.", "appeal to fear"),
    ("When we go to war, we kill children with drones and kill innocent civilians.", "appeal to fear"),
    ("If people smoke at home they hurt their family-members, the one they love, through passive-smoke. It was such a shame if this was not illegal.", "appeal to fear"),
    ("Yes, and that is why we do also have to censor the web. Think about all the brutality, which poisons our minds, all this porn that poisons the thoughts and relationships of our children.", "appeal to fear"),
    ("For little children it is already enough if one cheats. they might get drug addicts as well. You have the choice, save children or drug addicts.", "appeal to fear"),
    ("Do you want to have a Hitler again? No? Then this Topic is not arguable.", "appeal to fear"),
    ("Test them, otherwise our children might get drug addicts as well.", "appeal to fear"),
    ("We must keep Europe safe - these people are mostly muslim terrorist and they're going to overtake our country if we let them in!", "appeal to fear"),
    ("Me and my family have been living a peaceful life, so far. Everything is fine. I dont want sick black people to come and ruin my idyll.", "appeal to fear"),
    ("Imagine you have a little child that dies because a drunken driver causes an accident. Now tell me you dont want to have that forbidden.", "appeal to fear"),
    ("Co-ed schools... think about what this means. Young boys and girls together in a room, partly without supervision of a teacher. I would not want my daughter to get pregnant while school.", "appeal to fear"),
    ("If your mom or sister or friend and she was gang raped by a group of illegals wouldn't you want her to have the chance to have the baby taken care of?", "appeal to fear"),
    ("If you don't believe in Jesus Christ you'll suffer eternal damnation and torment in the afterlife.", "appeal to fear"),
    ("Look I know it seems pricey but think about the safety of your family. Without this home security system you leave your loved ones vulnerable to break-ins and potential harm.", "appeal to fear"),
    ("No. Do you really want our children to waste away in prison because they tried smoking weed at a party once? Just imagine your childs life being ruined by such a small mistake.", "appeal to fear"),
    ("But if Turkey joins the EU, poor Turks might invade our country and take away our jobs. DONT MAKE THAT HAPPEN!", "appeal to fear"),
    ("Marijuana ruins lives and should not be legal- it can leave children without their parents.", "appeal to fear"),
    ("We should go to war; remember the horrified people leaping from the World Trade Center? If we don't fight, that will happen again.", "appeal to fear"),
    ("They will bring the disease to our country and many people will die because of that.", "appeal to fear"),
    ("You might think you're healthy now but what if illness strikes? Taking this miracle drug could save your life.", "appeal to fear"),
    ("Consider the consequences of your vote. If you don't support this candidate you're risking the stability and future of our country.", "appeal to fear"),
    ("It's tempting to skip out on insurance but think about the worst-case scenario. Without coverage you could face devastating financial consequences in times of need.", "appeal to fear"),
    ("I know it's easy to dismiss alternative news sources but think about the information you might be missing. Without subscribing you're leaving yourself vulnerable to biased reporting.", "appeal to fear"),
    ("I understand your concerns but national security should be our top priority. Without this policy we're leaving ourselves vulnerable to potential threats.", "appeal to fear"),
    ("Sticking to a diet isn't easy. But think about the long-term health risks you're avoiding. Without it you're opening yourself up to serious diseases.", "appeal to fear"),
    ("Investing in this stock might seem risky but the alternative is financial disaster. Without it you could lose everything.", "appeal to fear"),
    ("If you don't buy this miracle anti-aging cream you'll look old and unattractive.", "appeal to fear"),
    ("If you don't enroll your child in this elite private school they'll fall behind academically and struggle to succeed in life.", "appeal to fear")
]


list_of_tuples = []
list_of_tuples += data_tuples_fear

datasets = [ad_hominem, ad_populum, appeal_to_anger, appeal_to_authority, appeal_to_nature, appeal_to_authority, appeal_to_nature, 
           appeal_to_pity, appeal_to_tradition, appeal_to_worse_problems, causal_oversimplifiation, 
           equivocation, fallacy_of_division, false_analogy, false_causality, false_dilemma, 
           hasty_generalization, nothing, slippery_slope, strawman]
for df in datasets:
   
    list_of_tuples += [tuple(x) for x in df.to_numpy()]

# Print the first few tuples
print(list_of_tuples[:5])
    

def contains_whole_word(large_string, word):
    pattern = rf'\b{re.escape(word)}\b'
    return bool(re.search(pattern, large_string))

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
examples = list_of_tuples
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
