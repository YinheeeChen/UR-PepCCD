import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # 1. 读取咱们刚刚生成的完整数据集
    csv_path = "/workspace/guest/cyh/workspace/PepCCD/dataset/PepFlow/pepflow_dataset.csv"
    print(f"读取全量数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 2. 按照 8:2 的比例随机划分 (test_size=0.2)
    # random_state=42 相当于设定随机种子，保证每次运行划分的结果都一样，方便复现！
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 3. 分别保存为 train 和 test 文件
    train_file = "/workspace/guest/cyh/workspace/PepCCD/dataset/PepFlow/pepflow_train.csv"
    test_file = "/workspace/guest/cyh/workspace/PepCCD/dataset/PepFlow/pepflow_test.csv"
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print("\n🎉 数据集划分完成！")
    print(f"总数据量: {len(df)} 条")
    print(f"-> 🏋️ 训练集 (Train) 已保存至 {train_file}，共 {len(train_df)} 条。")
    print(f"-> 🧪 测试集 (Test)  已保存至 {test_file}，共 {len(test_df)} 条。")

if __name__ == '__main__':
    main()