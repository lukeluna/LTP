from openai import OpenAI
import pandas as pd

### This is where you put your OpenAI API key for authorisation. 
client = OpenAI(
  api_key='123',
)


### This function takes a fallacy as input and calls gpt-3.5-turbo to detect what kind of fallacy it is.
### The choices are Appeal to Positive Emotion, Appeal to Anger, Appeal to Fear, Appeal to Pity, Appeal to Ridicule, and Appeal to Worse Problem
def detect_fallacy(fallacy):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    temperature=0.1,
    messages=[
        {"role": "system", "content": "You are a philosopher who specialises in analysing fallacies."},
        {"role": "user", "content": "What type of fallacy is this out of this list of fallacies? [Appeal to Positive Emotion, Appeal to Anger, Appeal to Fear, Appeal to Pity, Appeal to Ridicule, Appeal to Worse Problem] Fallacy: I cried so hard when i watched titanic, obviously it deserves these oscars."},
        {"role": "assistant", "content": "Fallacy of Positive Emotion"},
        {"role": "user", "content": "What type of fallacy is this out of this list of fallacies? [Appeal to Positive Emotion, Appeal to Anger, Appeal to Fear, Appeal to Pity, Appeal to Ridicule, Appeal to Worse Problem] Fallacy: " + fallacy}
    ]
    )
    return response.choices[0].message.content


### This function applies the detect_fallacy function to all fallacies given in a .csv file (with columns Sentence, Label).
def convert_all_fallacies(incsv,outcsv):
    df = pd.read_csv(incsv)

    df['Label'] = df['Sentence'].apply(detect_fallacy)

    df.to_csv(outcsv,index=False)



#print(detect_fallacy("yes they allow us to connect"))
convert_all_fallacies('appeal_to_emotion_argotario.csv','test.csv')