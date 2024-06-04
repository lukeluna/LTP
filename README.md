# LTP
Language Technology Project on Fallacy Detection


### github.csv

github.csv contains the concatenation of the fallacies contained in this github repo: https://github.com/Tariq60/fallacy-detection/tree/master . The labels are converted to the MAFALDA annotation scheme and format.

### argotario data

argotario_2018-01-15.csv contains the argotario data where the labels have been converted to the MAFALDA format and annotation scheme, apart from 'appeal to emotion', which is annotated as 'appeal to ?'.
argotario_in_mafalda_format.csv contains the same data, but the 'appeal to ?' items have been annotated semi-automatically.

### ElecDeb60to20 data

elecdeb60to20_final_fall.csv contains the elecdeb final fall data with the MAFALDA format labels and duplicates removed. It includes instances of 'appeal to emotion'. elecdeb60to20_appeal_to_emotion_labeled.csv was created by separating the 'appeal to emotion' instances from the previous dataset and generating labels using the classify_appeal_to_emotion_using_gpt.py code. Regex was performed on the data to retreive the final labels in MAFALDA format and any labels that were not easily identifiable were removed from the dataset. 


### Code

# get_mafalda_spans.py

This code was used to extract the data on which we tested our LLM prompts from the MAFALDA dataset. Spans annotated with fallacies are extracted and paired with their annotation in a .csv file. 
