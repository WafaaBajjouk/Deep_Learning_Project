import pandas as pd
from collections import Counter

file_path = 'merged_labeling_data.csv'
data = pd.read_csv(file_path)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

filtered_data = data[data['sentiment'].isin(['Neutral', 'Negative', 'Positive'])]

before_counts = Counter(filtered_data['sentiment'])
print("Sentiment Distribution Before Balancing:", before_counts)

min_count = min(before_counts.values())

balanced_data = (
    filtered_data.groupby('sentiment')
    .apply(lambda x: x.sample(min_count, random_state=42))
    .reset_index(drop=True)
)

after_counts = Counter(balanced_data['sentiment'])
print("Sentiment Distribution After Balancing:", after_counts)
