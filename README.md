# Language Technology Project Group 2

## Overview

This project is largely to prompt the Large Language Models of [Gemma](https://huggingface.co/google/gemma-7b-it), [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.3) and [Openchat](https://huggingface.co/openchat/openchat) with Chain-of-Thought Prompting, Chain-of-Thought Self-Consistency Prompting, Tree-Of-Thought Prompting and Rephrase and Respond Prompting.

## Dataset

### Handmade Dataset / HEFA - High-quality Examples of Fallacious Arguments
*File paths relative to* `data/handmade`

HEFA, a dataset containing fallacious arguments labeled by their respesctive fallacy type.\ 
We use the same fallacy taxonomy as [MAFALDA](https://arxiv.org/abs/2311.09761). 

More information can go to https://github.com/rosakun/Fallacies

### MAFALDA
*File paths relative to* `data/mafalda`
TBD

### github.csv

github.csv contains the concatenation of the fallacies contained in this github repo: https://github.com/Tariq60/fallacy-detection/tree/master . The labels are converted to the MAFALDA annotation scheme and format.

### argotario data

argotario_2018-01-15.csv contains the argotario data where the labels have been converted to the MAFALDA format and annotation scheme, apart from 'appeal to emotion', which is annotated as 'appeal to ?'.
argotario_in_mafalda_format.csv contains the same data, but the 'appeal to ?' items have been annotated semi-automatically.

### ElecDeb60to20 data

elecdeb60to20_final_fall.csv contains the elecdeb final fall data with the MAFALDA format labels and duplicates removed. It includes instances of 'appeal to emotion'. elecdeb60to20_appeal_to_emotion_labeled.csv was created by separating the 'appeal to emotion' instances from the previous dataset and generating labels using the classify_appeal_to_emotion_using_gpt.py code. Regex was performed on the data to retreive the final labels in MAFALDA format and any labels that were not easily identifiable were removed from the dataset. 

## Requirements
Install [Ollama](https://github.com/ollama/ollama) in line with the operating system of the current environment. Make sure it is running in the host computer. If not, run `ollama serve` first.

Install [Gemma](https://ollama.com/library/gemma), [Mistral](https://ollama.com/library/mistral), [Openchat](https://ollama.com/library/openchat) by pulling them with the command:

```
ollama pull gemma
ollama pull mistral
ollama pull openchat
```

Run `pip install -r requirements.txt` for the file `requirements.txt` included on the repo.

## Preprocessing
*File paths relative to* `src/preprocessing`

## Prompting

Prompt can be run by running

```
python3 src/main.py
```

on the main directory.

The prompts use are in folder `prompts`

- few_shot_cot_update (Complete and Main version of CoT and CoT-SC): `prompts/few_shot_cot_update.txt`
- few_shot_cot : `prompts/few_shot_cot.txt`
- RaR_old (Main version of RaR): `prompts/RaR_old.txt`
- RaR_new : `prompts/RaR_new.txt`
- RaR_all_new : `prompts/RaR_all_new.txt`
- tot_extra_shortened : `prompts/tot_extra_shortened.txt`
- tot_with_examples (Main version of ToT) : `prompts/tot_with_examples.txt`
- tot_with_explanations : `prompts/tot_with_explanations.txt`
- zero_shot_tot_2 : `prompts/zero_shot_tot_2.txt`
- zero_shot_tot : `prompts/zero_shot_tot.txt`

## Postprocessing
*File paths relative to* `src/postprocessing`

