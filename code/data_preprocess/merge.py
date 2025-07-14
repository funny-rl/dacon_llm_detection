import os 
import pandas as pd


def merge_csv_files(file_paths: list, output_filename: str = "merged_output.csv") -> pd.DataFrame:
    all_dataframes = []
    print("🚀 CSV 파일 병합을 시작합니다...")

    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다. 이 파일은 건너뜝니다.")
            continue

        try:
            df = pd.read_csv(file_path)
            all_dataframes.append(df)
            print(f"✅ [{i+1}/{len(file_paths)}] '{file_path}' 로드 완료. (총 {len(df)} 행)")
        except Exception as e:
            print(f"❌ 오류: '{file_path}' 로드 중 문제 발생: {e}. 이 파일은 건너뜝니다.")

    if not all_dataframes:
        print("⚠️ 병합할 유효한 CSV 파일이 없습니다. 작업을 종료합니다.")
        return pd.DataFrame() 

    
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✨ 모든 파일 통합 완료. 중복 제거 전 총 {len(merged_df)} 행.")

    initial_rows = len(merged_df)
    merged_df.drop_duplicates(inplace=True)
    removed_rows = initial_rows - len(merged_df)

    if removed_rows > 0:
        print(f"🧹 중복된 행 {removed_rows}개를 제거했습니다.")
    else:
        print("👍 중복된 행이 없습니다.")

    print(f"✅ 최종 병합된 데이터 총 {len(merged_df)} 행.")

    try:
        merged_df.to_csv(output_filename, index=False)
        print(f"\n🎉 병합된 데이터가 '{output_filename}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 오류: '{output_filename}' 저장 중 문제 발생: {e}")
        
    return merged_df

if __name__ == "__main__":
    csv_files_to_merge = [
        "./data/train_random_30000_augmented_1.csv",
        "./data/train_random_30000_augmented_2.csv",
        "./data/train_random_30000_augmented_3.csv",
        "./data/train_random_30000_augmented_4.csv",
        "./data/train.csv",
    ]
    #csv_files_to_merge = [
    #     "./data/split_augmented.csv",
    #     "./data/no_split.csv",
    # ]
    
    # csv_files_to_merge = [
    #     "./data/total_data.csv",
    #     "./data/filtered_train.csv",
    # ]
    
    #output_csv_name = "./data/total_data.csv"
    #output_csv_name = "./data/split_augmented.csv"
    output_csv_name = "./data/total_data_2.csv"
    
    final_merged_df = merge_csv_files(csv_files_to_merge, output_csv_name)

    if not final_merged_df.empty:
        print(f"\n📝 '{output_csv_name}' 파일의 상위 5행:\n{final_merged_df.head()}")
    else:
        print("\n최종 병합된 데이터프레임이 비어 있습니다.")