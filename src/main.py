import time
import ollama
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
from tqdm import tqdm
import sys
import random
from collections import Counter

DATA_BASE_FILEPATH = 'data'
DATA_FILEPATH = '{}/handmade'.format(DATA_BASE_FILEPATH)
PROMPT_FILEPATH = 'prompts'

filepath_zs_tot = '{}/zero_shot_tot.txt'.format(PROMPT_FILEPATH)
filepath_tot = '{}/tot_with_examples.txt'.format(PROMPT_FILEPATH)
filepath_rar_1 = '{}/RaR_old.txt'.format(PROMPT_FILEPATH)
filepath_rar_2 = '{}/RaR_new.txt'.format(PROMPT_FILEPATH)
filepath_rar_3 = '{}/RaR_all_new.txt'.format(PROMPT_FILEPATH)
filepath_cot = '{}/few_shot_cot_update.txt'.format(PROMPT_FILEPATH)

ad_hominem = pd.read_csv('{}/ad_hominem_final.csv'.format(DATA_FILEPATH))
ad_populum = pd.read_csv('{}/ad_populum_final.csv'.format(DATA_FILEPATH))
appeal_to_anger = pd.read_csv('{}/appeal_to_anger_final.csv'.format(DATA_FILEPATH))
appeal_to_authority = pd.read_csv('{}/appeal_to_authority_final.csv'.format(DATA_FILEPATH))
appeal_to_fear = pd.read_csv('{}/appeal_to_fear_final.csv'.format(DATA_FILEPATH))
appeal_to_nature = pd.read_csv('{}/appeal_to_nature_final.csv'.format(DATA_FILEPATH))
appeal_to_pity = pd.read_csv('{}/appeal_to_pity_final.csv'.format(DATA_FILEPATH))
appeal_to_ridicule = pd.read_csv('{}/appeal_to_ridicule_final.csv'.format(DATA_FILEPATH))
appeal_to_tradition = pd.read_csv('{}/appeal_to_tradition_final.csv'.format(DATA_FILEPATH))
appeal_to_worse_problems = pd.read_csv('{}/appeal_to_worse_problems_final.csv'.format(DATA_FILEPATH))
causal_oversimplifiation = pd.read_csv('{}/causal_oversimplification_final.csv'.format(DATA_FILEPATH))
equivocation = pd.read_csv('{}/equivocation_final.csv'.format(DATA_FILEPATH))
fallacy_of_division = pd.read_csv('{}/fallacy_of_division_final.csv'.format(DATA_FILEPATH))
false_analogy = pd.read_csv('{}/false_analogy_final.csv'.format(DATA_FILEPATH))
false_causality = pd.read_csv('{}/false_causality_final.csv'.format(DATA_FILEPATH))
false_dilemma = pd.read_csv('{}/false_dilemma_final.csv'.format(DATA_FILEPATH))
hasty_generalization = pd.read_csv('{}/hasty_generalization_final.csv'.format(DATA_FILEPATH))
nothing = pd.read_csv('{}/nothing_final.csv'.format(DATA_FILEPATH))
slippery_slope = pd.read_csv('{}/slippery_slope_final.csv'.format(DATA_FILEPATH))
strawman = pd.read_csv('{}/strawman_final.csv'.format(DATA_FILEPATH))
circular_reasoning = pd.read_csv('{}/circular_reasoning.csv'.format(DATA_FILEPATH))
tu_quoque = pd.read_csv('{}/tu_quoque.csv'.format(DATA_FILEPATH))

mafalda = pd.read_csv('{}/mafalda/mafalda_gold_standard_by_span.csv'.format(DATA_BASE_FILEPATH))


def prepare_dataset(is_mafalda=True):
    list_of_tuples = []
    nr_of_samples = 150
    sample = False
    
    if is_mafalda:
        datasets = [mafalda]
    else:
        datasets = [ad_hominem, ad_populum, appeal_to_anger, appeal_to_authority, appeal_to_fear, appeal_to_nature,
                    appeal_to_pity, appeal_to_ridicule, appeal_to_tradition, appeal_to_worse_problems, causal_oversimplifiation,
                    circular_reasoning, tu_quoque, #guilt_by_association,
                    equivocation, fallacy_of_division, false_analogy, false_causality, false_dilemma,
                    hasty_generalization, nothing, slippery_slope, strawman]
    
    for dataset in datasets:
        if sample:
            if len(dataset) >= nr_of_samples:
                # Sample 15 entries if the dataset is large enough
                sampled = dataset.sample(n=nr_of_samples, random_state=random.randint(1, 100))  # Change seed for true randomness
            else:
                # If the dataset has less than 15 entries, take all available data
                sampled = dataset
        else:
            sampled = dataset
    
        # Convert the sampled data to tuples and add to the list
        list_of_tuples.extend(sampled.itertuples(index=False, name=None))

    return list_of_tuples

def ollama_prompt(model_id, message_content):
    response = ollama.chat(model=model_id, messages=[
        {
            'role': 'user',
            'content': message_content
        }
    ])
    
    return response['message']['content']

def contains_whole_word(large_string, word):
    pattern = rf'\b{re.escape(word)}\b'
    return bool(re.search(pattern, large_string))

def rephrase_question(model_id, question):
    rephrase_prompt = f"Rephrase and expand the following question to improve clarity and detail: {question}"
    rephrase_response = ollama.chat(model_id, messages=[
        {
            'role': 'user',
            'content': rephrase_prompt
        }
    ])
    rephrased_question = rephrase_response['message']['content'].strip()
    return rephrased_question

def get_response(model_id, rephrased_question):
    response_prompt = rephrased_question
    response = ollama.chat(model_id, messages=[
        {
            'role': 'user',
            'content': response_prompt
        }
    ])
    return response['message']['content'].strip().lower()

def prompt(model_id, message_base_content, is_mafalda=True,  selfconsistency=False, rar=False):
    test_results = []
    results = pd.DataFrame()
    start_time = time.time() 

    examples = prepare_dataset(is_mafalda=is_mafalda)
    
    for text, expected_label in tqdm(examples):
        message_content = message_base_content.format(text)

        if selfconsistency:
            loop = 3
            labels = [] 
            test_passeds = []
            for i in range(loop):
                subresponse = ollama_prompt(model_id, message_content)
                label = subresponse.strip()
                label = label.lower()
                labels.append(label)
                test_passeds.append(contains_whole_word(label, expected_label))

            c = Counter(test_passeds)
            test_passed, _ = c.most_common()[0]
            test_results.append(test_passed)
    
            results = results._append({
                'text': text,
                'expected_label': expected_label,
                'actual_label1': labels[0],
                'actual_label2': labels[1],
                'actual_label3': labels[2],
                'result': test_passed
            }, ignore_index=True)
        
        else:
            if rar:
                rephrased_question = rephrase_question(model_id, message_content)
                actual_label = get_response(model_id, rephrased_question)
                test_passed = contains_whole_word(actual_label, expected_label)
            else:
                response = ollama_prompt(model_id, message_content)
                actual_label = response.strip()
                actual_label = actual_label.lower()
            
            test_passed = contains_whole_word(actual_label, expected_label)
            test_results.append(test_passed)
    
            results = results._append({
                'text': text,
                'expected_label': expected_label,
                'actual_label': actual_label,
                'result': test_passed
            }, ignore_index=True)
            
    end_time = time.time()
    accuracy = sum(test_results) / len(test_results)
    f1 = f1_score([True]*len(test_results), test_results, average='weighted')
    precision = precision_score([True]*len(test_results), test_results, average='weighted')
    recall = recall_score([True]*len(test_results), test_results, average='weighted')

    time_taken = end_time - start_time

    return accuracy, f1, precision, recall, time_taken, results

def run_prompt(message_base_content, title="Tot", is_mafalda=True, selfconsistency=False, rar=False):
    for model in ["mistral", "gemma", "openchat"]:
        accuracy, f1, precision, recall, time_taken, results = prompt(model, message_base_content, is_mafalda, selfconsistency, rar)
    
        print('Accuracy {} {}: '.format(title, model), accuracy)
        print('F1 score {} {}:'.format(title, model), f1)
        print('Precision {} {}: '.format(title, model), precision)
        print('Recall {} {}: '.format(title, model), recall)
        
        print(f'Time Taken: {time_taken:.2f} seconds\n')
        
        results.to_csv("results/results-{}-{}.csv".format(title, model), sep=",")

def main():
    with open(filepath_zs_tot, 'r') as file:
        message_base_content_zs_tot = file.read()
    file.close()

    with open(filepath_tot, 'r') as file:
        message_base_content_tot = file.read()
    file.close()

    with open(filepath_cot, 'r') as file:
        message_base_content_cot = file.read()
    file.close()

    with open(filepath_rar_1, 'r') as file:
        message_base_content_rar_1 = file.read()
    file.close()

    with open(filepath_rar_2, 'r') as file:
        message_base_content_rar_2 = file.read()
    file.close()

    with open(filepath_rar_3, 'r') as file:
        message_base_content_rar_3 = file.read()
    file.close()

    run_prompt(message_base_content_cot, title="cot-mafalda-spans", is_mafalda=True, selfconsistency=False, rar=False)
    run_prompt(message_base_content_cot, title="cot-sc-mafalda-spans", is_mafalda=True, selfconsistency=True, rar=False)
    run_prompt(message_base_content_zs_tot, title="zs-tot-mafalda-spans", is_mafalda=True, selfconsistency=False, rar=False)
    run_prompt(message_base_content_tot, title="tot-mafalda-spans", is_mafalda=True, selfconsistency=False, rar=False)
    run_prompt(message_base_content_rar_3, title="rar-3-mafalda-spans", is_mafalda=True, selfconsistency=False, rar=True)
    run_prompt(message_base_content_rar_2, title="rar-2-mafalda-spans", is_mafalda=True, selfconsistency=False, rar=True)
    run_prompt(message_base_content_rar_1, title="rar-1-mafalda-spans", is_mafalda=True, selfconsistency=False, rar=True)

if __name__ == "__main__":
    main()
