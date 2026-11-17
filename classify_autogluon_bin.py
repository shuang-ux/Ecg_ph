
from .bjutecgdataset import load_ecg_feature
from .figure_tools import *
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import GroupShuffleSplit,ShuffleSplit
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


def classify_autogluon(csv_file,savepath = './mid/autogluon_results',label_column = 'label',eval_metric='f1',test_size=0.2,random_state=42,hyperparameters = {
    'GBM': {},   # LightGBM
    'CAT': {},   # CatBoost
    'RF': {},    # RandomForest
    'XGB': {}    # XGBoost
    # 'NN' 
}):
    
    os.makedirs(savepath,exist_ok=True)
    



    test_size = 0.2
    random_state = 42

    if isinstance(csv_file,str):
        task_df = pd.read_csv(csv_file)    
    elif isinstance(csv_file,pd.DataFrame):
        task_df = csv_file
    else:
        raise ValueError('csv_file must be a str or a pd.DataFrame')
    
    #print("1️⃣ task_df:", len(task_df))
    task_df = task_df.dropna(subset=label_column)
    #print("2️⃣ 删除 label 为空后:", len(task_df))

    X = load_ecg_feature(task_df['record'],task_df['db'],alg='gri')
    # print("3️⃣ ECG 特征 X:", len(X))
    # print("X 唯一 record 数:", X['record'].nunique())
    # records_before = set(task_df['record'])
    # records_after = set(X['record'])

    # missing = records_before - records_after
    # extra = records_after - records_before

    # print("task_df 中有但 X 中没有:", len(missing))
    # print("X 中有但 task_df 中没有:", len(extra))

    # if 'subject_id' not in task_df.columns:
    #     df_feature_with_label = pd.merge(task_df[['record',label_column,'age','gender']], X, on='record', how='left')
    #     gss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    # else:
    #     df_feature_with_label = pd.merge(task_df[['record',label_column,'age','gender','subject_id']], X, on='record', how='left')
    #     gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    if 'subject_id' not in task_df.columns:
        df_feature_with_label = pd.merge(task_df[['record',label_column]], X, on='record', how='left')
        gss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    else:
        df_feature_with_label = pd.merge(task_df[['record',label_column,'subject_id']], X, on='record', how='left')
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    #print("4️⃣ merge 后:", len(df_feature_with_label))
    df_feature_with_label.rename(columns={label_column:'label'}, inplace=True)
    #print("5️⃣ rename 后:", len(df_feature_with_label))
    df_feature_with_label = df_feature_with_label.dropna(subset=['conclusion'])
    #print("6️⃣ 删除 conclusion 为空后:", len(df_feature_with_label))
    df_feature_with_label = df_feature_with_label.dropna(subset=['label'])
    #print("7️⃣ 删除 label 为空后:", len(df_feature_with_label))
    df_feature_with_label = df_feature_with_label.drop(columns=['conclusion','rpos','code','nBeats'])
    #print("8️⃣ 删除列后（行数不变）:", len(df_feature_with_label))


    train_idx, test_idx = next(gss.split(df_feature_with_label, df_feature_with_label['label'], df_feature_with_label['subject_id']))
    train_data = df_feature_with_label.iloc[train_idx]
    test_data = df_feature_with_label.iloc[test_idx]

    # train_records = df_feature_with_label.iloc[train_idx]['record']
    # test_records  = df_feature_with_label.iloc[test_idx]['record']

    # train_raw = task_df[task_df['record'].isin(train_records)]
    # test_raw  = task_df[task_df['record'].isin(test_records)]

    train_data = train_data.drop(columns=['record','subject_id'])
    test_data = test_data.drop(columns=['record','subject_id'])
    # # print(f"测试集数量: {len(test_data)}")

    # # 保存
    # train_raw.to_csv(f"{savepath}/train_raw.csv", index=False)
    # test_raw.to_csv(f"{savepath}/test_raw.csv", index=False)

    print("\n>>> 合并 ECG 特征后的数据统计")
    print(f"总样本数: {len(df_feature_with_label)}")

    for split_name, split_df in {
        "train": train_data,
        "test": test_data
    }.items():
        print(f"\n{split_name} 集:")
        print(f"  样本数: {len(split_df)}")
        # print(f"  性别分布:\n{split_df['gender'].value_counts(dropna=False)}")
        print(f"  标签分布:\n{split_df['label'].value_counts().sort_index()}")
    
    # 判断 train_data 的 label 列是否为多分类
    num_classes = df_feature_with_label['label'].nunique()
    # 定义允许的指标（AutoGluon 支持）
    valid_binary_metrics = ['roc_auc', 'f1', 'f1_macro', 'f1_weighted', 'accuracy', 'balanced_accuracy', 'log_loss']
    valid_multiclass_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'log_loss', 'softlogloss']

    # 自动选择 metric
    if num_classes == 2:     # 二分类
        if eval_metric in valid_binary_metrics:
            eval_metric = eval_metric
        else:
            eval_metric = 'roc_auc'       # 默认
    else:                    # 多分类
        if eval_metric in valid_multiclass_metrics:
            eval_metric = eval_metric
        else:
            eval_metric = 'balanced_accuracy' 


    predictor = TabularPredictor(label='label',
                                eval_metric=eval_metric ,
                                path=savepath
                                ).fit(
                                    train_data=train_data,
                                    hyperparameters = hyperparameters
                                    
                                    )
    # predictor.set_decision_threshold(0.5)
    print(f"查看当前阈值: {predictor.decision_threshold}")
    # Leaderboard
    # -------------------------
    leaderboard = predictor.leaderboard(test_data, silent=True)
    leaderboard[['model', 'score_val', 'score_test']].to_csv(
        f'{savepath}/leaderboard.csv', index=False
    )

    # -------------------------
    # 预测类别和概率
    # -------------------------
    y_pred = predictor.predict(test_data)
    proba = predictor.predict_proba(test_data)     # DF, shape = (N, K)
    y_true = test_data['label']

    # 自动判断类别数
    num_classes = proba.shape[1]

    # -------------------------
    # Metrics
    # -------------------------
    metrics_output = {}

    # 分类报告
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report)
    df_report.to_csv(f"{savepath}/metrics_report.csv")

    metrics_output["accuracy"] = report["accuracy"]

    # -------- 二分类单独处理 ROC --------
    if num_classes == 2:
        # AutoGluon 的 predict_proba 二分类输出两列：class0_prob, class1_prob
        y_score = proba.iloc[:, 1]   # 正类概率
        auc = roc_auc_score(y_true, y_score)
        metrics_output["AUC"] = auc

    # -------- 多分类（≥3 类） --------
    else:
        try:
            auc_macro = roc_auc_score(y_true, proba, multi_class='ovr', average='macro')
            auc_micro = roc_auc_score(y_true, proba, multi_class='ovr', average='micro')
            metrics_output["AUC_macro"] = auc_macro
            metrics_output["AUC_micro"] = auc_micro
        except:
            pass

    pd.DataFrame([metrics_output]).to_csv(
        f"{savepath}/metrics_summary.csv", index=False
    )
    
    
        # ----------------------
    # 混淆矩阵
    # ----------------------
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, 
                        index=[f"true_{c}" for c in predictor.class_labels],
                        columns=[f"pred_{c}" for c in predictor.class_labels])
    df_cm.to_csv(f"{savepath}/confusion_matrix.csv", index=True)

    # 补充NPV
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics_output["NPV"] = npv

    pd.DataFrame([metrics_output]).to_csv(
        f"{savepath}/metrics_summary.csv", index=False
)

    # -------------------------
    # 特征重要性
    # -------------------------
    importance = predictor.feature_importance(test_data)
    importance.to_csv(f'{savepath}/feature_importance.csv')

    # -------------------------
    # 统一输出 y_true / y_pred / 概率
    # -------------------------

    print(f"proba 索引: {proba.index[:5]}")
    print(f"X_test 索引: {test_data.index[:5]}")
    print(f"y_true 索引: {y_true.index[:5]}")

    # 确认所有索引相同
    assert all(proba.index == test_data.index), "proba 和 X_test 索引不匹配！"
    assert all(proba.index == y_true.index), "proba 和 y_true 索引不匹配！"

    df_out = proba.copy()
    df_out["y_true"] = y_true.values
    df_out["y_pred"] = y_pred.values

    # 添加 age 和 gender 列
    # df_out["age"] = test_data["age"].values  
    # df_out["gender"] = test_data["gender"].values  

    df_out.to_csv(f'{savepath}/y_true_score.csv', index=False)
    
    return train_idx, test_idx
