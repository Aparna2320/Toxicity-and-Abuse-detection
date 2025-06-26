import pandas as pd

df = pd.read_csv("train.csv")

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def get_reasons(row):
    return [cat for cat in categories if row[cat] == 1]

def get_label(reasons):
    return 'safe' if len(reasons) == 0 else 'toxic'

df['reasons'] = df.apply(get_reasons, axis=1)
df['label'] = df['reasons'].apply(get_label)

df_cleaned = df[['comment_text', 'label', 'reasons']].rename(columns={'comment_text': 'text'})

safe = df_cleaned[df_cleaned['label'] == 'safe'].sample(500, random_state=42)
toxic = df_cleaned[df_cleaned['label'] == 'toxic'].sample(500, random_state=42)
final_df = pd.concat([safe, toxic]).sample(frac=1).reset_index(drop=True)

final_df.to_csv("cleaned_toxic_dataset.csv", index=False)
print("✅ Dataset is ready! Saved as cleaned_toxic_dataset.csv")

