import pandas as pd

if __name__ == "__main__":
    """Remove the file that causes problems with librosa and save the csv again"""
    csv_path = 'Data/features_30_sec.csv'
    df = pd.read_csv(csv_path)
    df = df.drop(df[df['filename'] == 'jazz.00054.wav'].index)
    df.to_csv('Data/features_30_sec.csv', index=False)

    print('Removed jazz.00054.wav, now you can do stuff in peace:)')
