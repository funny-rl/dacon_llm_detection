import pandas as pd
from sklearn.model_selection import train_test_split
import os 

input_csv_path = "./data/total_data_2.csv"
train_output_path = "../data/aug30000/train.csv"
valid_output_path = "../data/aug30000/valid.csv"

print("🚀 데이터셋 로드 및 분할을 시작합니다...")

try:
    df_total = pd.read_csv(input_csv_path)
    print(f"✅ '{input_csv_path}' 파일 로드 완료. (총 {len(df_total)} 행)")
except FileNotFoundError:
    print(f"❌ 오류: '{input_csv_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()
except Exception as e:
    print(f"❌ 오류: '{input_csv_path}' 로드 중 문제 발생: {e}")
    exit()

train_df, valid_df = train_test_split(
    df_total,
    test_size=0.1,         
    random_state=42,     
    shuffle=True,          
    stratify=df_total['label'] 
)

print(f"\n✨ 데이터 분할 완료! 데이터는 무작위로 섞인 후 분할되었습니다.")
print(f"훈련 세트 ({os.path.basename(train_output_path)}): {len(train_df)} 행")
print(f"검증 세트 ({os.path.basename(valid_output_path)}): {len(valid_df)} 행")

# 분할된 데이터의 라벨 비율 확인 (선택 사항)
print("\n--- 분할된 데이터셋의 라벨 비율 확인 ---")
if 'label' in train_df.columns:
    print("훈련 세트 라벨 비율:\n", train_df['label'].value_counts(normalize=True) * 100)
    print("\n검증 세트 라벨 비율:\n", valid_df['label'].value_counts(normalize=True) * 100)
else:
    print("⚠️ 'label' 컬럼이 분할된 데이터프레임에 존재하지 않아 라벨 비율을 확인할 수 없습니다.")

try:
    train_df.to_csv(train_output_path, index=False)
    print(f"\n🎉 훈련 세트가 '{train_output_path}' 파일에 성공적으로 저장되었습니다.")
except Exception as e:
    print(f"❌ 오류: '{train_output_path}' 저장 중 문제 발생: {e}")

try:
    valid_df.to_csv(valid_output_path, index=False)
    print(f"🎉 검증 세트가 '{valid_output_path}' 파일에 성공적으로 저장되었습니다.")
except Exception as e:
    print(f"❌ 오류: '{valid_output_path}' 저장 중 문제 발생: {e}")

print("\n🚀 모든 과정이 완료되었습니다.")