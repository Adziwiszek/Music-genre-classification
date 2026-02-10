from feature_extraction_pipeline import build_feature_pipeline
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    csv_path = "Data/features_custom.csv"
    genres = [
      "blues",
      "classical",
      "country",
      "disco",
      "hiphop",
      "jazz",
      "metal",
      "pop",
      "reggae",
      "rock",
    ]
    example_files = [
      Path("Data") / "genres_original" / genre / f"{genre}.{i:05}.wav"
      for genre in genres
      for i in range(0, 100)
      if i != 54 and genre != 'jazz'
    ]

    pipeline = build_feature_pipeline()

    X = pipeline.fit_transform(example_files)

    columns = []
    for old_c in pipeline.get_feature_names_out():
        if old_c.startswith('filename__'):
            columns.append('filename')
        elif old_c.startswith('genre__'):
            columns.append('genre')
        else:
            columns.append(old_c.split('__')[-1])
    
    df = pd.DataFrame(X, columns=columns)
    print(f"Extracted {df.shape[1]} features from {df.shape[0]} files.")
    df.to_csv(csv_path, index=False)
    print(f'saved to {csv_path}')
