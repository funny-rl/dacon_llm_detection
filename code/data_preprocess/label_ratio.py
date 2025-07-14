import pandas as pd

def data_dist(df: pd.DataFrame) ->None:
    label_counts = df['label'].value_counts()
    label_proportions = df['label'].value_counts(normalize=True) * 100

    print("Total data: ", len(df), "rows")
    
    for label_val in sorted(label_counts.index):
        count = label_counts[label_val]
        proportion = label_proportions[label_val]
        print(f"Label {label_val}: {count} ({proportion:.2f}%)")

    if 'text' in df.columns and 'label' in df.columns:
        df['text_length'] = df['text'].astype(str).apply(len)

    label_0_stats = df[df['label'] == 0]['text_length'].agg(['mean', 'std'])
    label_1_stats = df[df['label'] == 1]['text_length'].agg(['mean', 'std'])

    print(f"Label 0 text length - mean: {label_0_stats['mean']:.2f}, std: {label_0_stats['std']:.2f}")
    print(f"Label 1 text length - mean: {label_1_stats['mean']:.2f}, std: {label_1_stats['std']:.2f}")

if __name__ == "__main__":
    csv_path = "./data/split_augmented.csv"
    print("📊 Data Analysis...\n")
    df = pd.read_csv(csv_path)
    print(f"CSV path:{csv_path}\n")   
    data_dist(df)