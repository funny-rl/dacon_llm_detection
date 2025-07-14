import pandas as pd

# 파일 경로 설정
train_path = "./data/train.csv"
no_split_path = "./data/no_split.csv"
output_path = "./data/filtered_train.csv"

# CSV 파일 불러오기
train_df = pd.read_csv(train_path)
no_split_df = pd.read_csv(no_split_path)

# no_split에 있는 title 집합 생성
no_split_titles = set(no_split_df['title'].astype(str))

# 조건에 맞는 row 필터링
filtered_df = train_df[
    (~train_df['title'].astype(str).isin(no_split_titles)) & (train_df['label'] == 1)
]

# 결과 저장
filtered_df.to_csv(output_path, index=False)
print(f"✅ 필터링된 {len(filtered_df)}개의 row를 {output_path}에 저장했습니다.")