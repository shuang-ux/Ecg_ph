# 肺动脉高压（Pulmonary Hypertension, PH）ECG 分类

基于心电图（ECG）特征 + AutoGluon 自动机器学习，对**肺动脉高压（PH）**进行分级/二分类建模。

- 训练核心：`classify_autogluon_bin.py`（`classify_autogluon` 函数）＋ `step2_automodel_ph.py`（训练主流程）
- 外部验证：`step4_ExternalValSet_ph.py`
- 特征来源：ECG 特征表格（即 `X`），例如 `qinghaiai_gri_result_Ph.xlsx`

---

## 1. 整体流程

```
① 加载特征 X
   load_ecg_feature(record, db, alg='gri')  ← 从 ECG 特征表格读取

② 准备标签 y
   标签文件(record, subject_id, age, gender, 肺动脉压类型/数值)
   → 生成 ph_level → label1 / label2 / label3

③ 合并 + 切分
   按 record 合并 X 与标签；按 subject_id 分组切分 train/test（防泄漏）

④ 训练
   AutoGluon TabularPredictor（GBM / CAT / RF / XGB）

⑤ 验证
   step4 加载外部验证集 + 已训练模型 → 预测 + 评估指标
```

一句话：**X 来自特征表格，标签来自标签文件，两者按 `record` 关联，交给 AutoGluon 训练，再用外部集验证。**

---

## 2. 数据准备

### 2.1 特征表格 `X`

即 `qinghaiai_gri_result_Ph.xlsx` 这种形式。每一行是一条 ECG 记录，由 `load_ecg_feature` 读入，作为模型的输入特征 `X`。

主要列：

| 列 | 含义 | 是否用作特征 |
|---|---|---|
| `record` | 心电记录 ID（**与标签文件关联的键**） | 仅用于关联，训练时删除 |
| `conclusion` | ECG 诊断结论文本 | ❌ 训练时删除 |
| `code` | 诊断编码 | ❌ 训练时删除 |
| `rpos` | R 峰位置列表 | ❌ 训练时删除 |
| `nBeats` | 心搏数 | ❌ 训练时删除 |
| `VentricularRate`, `PQInterval`, `PDuration`, `QRSDuration`, `QTInterval`, `QTCInterval`, `PAxis`, `RAxis`, `TAxis` | 心率、间期、时限、电轴等基本参数 | ✅ |
| `P/P1/Q/R/R1/S/S1/J/JX/JXE/T/T1` × `Amplitude/Duration` × 12 导联（I, II, III, aVR, aVL, aVF, V1–V6） | 各波形幅度与时限 | ✅ |

> 即：除 `record` 外的数值列几乎全部作为特征；`conclusion / code / rpos / nBeats` 在训练前会被 `classify_autogluon` 自动剔除（见 `classify_autogluon_bin.py` 中 `drop(columns=['conclusion','rpos','code','nBeats'])`）。

### 2.2 标签文件

每行一条记录，至少包含：

| 列 | 含义 |
|---|---|
| `record` | 心电记录 ID（与特征表对应） |
| `subject_id` | 受试者 ID（用于分组切分，防止同一受试者泄漏到训练/测试两侧） |
| `age` / `gender` | 年龄 / 性别（元数据） |
| `肺动脉压类型` | 收缩 / 平均 / 平均压 等 |
| `肺动脉压数值` | 压力值（mmHg） |

> 标签由 `肺动脉压类型` 与 `肺动脉压数值` 计算得到（见下节）。

---

## 3. 标签定义

`step2_automodel_ph.py` 中的 `label_ph()` 先把肺动脉压分成 4 级（仅当 `肺动脉压类型 == "收缩"` 时才参与，平均压/平均会被过滤）：

| `ph_level` | 含义 | 收缩压范围（mmHg） |
|---|---|---|
| 0 | 正常 | < 35 |
| 1 | 轻度 | 35 – 50 |
| 2 | 中度 | 51 – 70 |
| 3 | 重度 | > 70 |

再由 `create_labels()` 派生出 3 个可训练的标签：

| 标签 | 含义 | 类型 |
|---|---|---|
| `label1` | 中度及以上（`ph_level > 1`） | 二分类 |
| `label2` | 轻度及以上（`ph_level > 0`） | 二分类 |
| `label3` | 0=正常 / 1=轻度 / 2=中度及以上 | 三分类 |

另外 `mode='exclude_mid_range'` 会排除 35–50 mmHg 的中间区间，只做「正常 vs 中高度肺动脉高压」的二分类 `label1`。

---

## 4. 三个脚本说明

### 4.1 `classify_autogluon_bin.py` —— 训练核心函数

对外暴露 `classify_autogluon(csv_file, savepath, label_column, eval_metric, test_size, random_state, hyperparameters)`，它做了：

1. 读取任务数据（`record` + 标签列 + `subject_id`）
2. 调用 `load_ecg_feature(records, db, alg='gri')` **加载特征 X**
3. 按 `record` 合并特征与标签，剔除 `conclusion / rpos / code / nBeats`
4. **切分**：若存在 `subject_id` 用 `GroupShuffleSplit`（按受试者分组），否则用 `ShuffleSplit`
5. 用 `TabularPredictor` 训练（默认 `GBM / CAT / RF / XGB` 四个模型）
6. 输出：leaderboard、分类报告、混淆矩阵、特征重要性、`y_true_score.csv`、AUC 等


### 4.2 `step2_automodel_ph.py` —— 训练主流程

- `load_data()`：加载标签文件（Excel，`dtype={'subject_id': str}`）
- `label_ph()` / `create_labels()`：生成标签
- `main_ph_workflow(mode, labels_to_train)`：对每个标签循环调用 `classify_autogluon` 训练

文件里有两个「cell」：第一个「心电 + 年龄性别」、第二个「不含年龄性别」，逻辑相同，区别只在 `df_classify` 是否带上 `age/gender`（`classify_autogluon` 内部当前都会剔除）。

### 4.3 `step4_ExternalValSet_ph.py` —— 外部验证

对外暴露 `evaluate_external_validation(model_dir, external_file, label_name)`，它做了：

1. 加载外部验证集（支持 `.pkl / .xlsx / .csv`）
2. 用与训练一致的逻辑生成 `ph_level` 和 `label1/label2/label3`
3. 加载特征 X、合并、清洗（并打印「数据丢弃原因」统计）
4. `TabularPredictor.load(model_dir)` 加载已训练模型
5. 预测，输出准确率 / AUC / 分类报告 / 混淆矩阵 / 预测结果

结果保存到 `{model_dir}_external_results/`。

---

## 5. 使用方法

> ⚠️ 代码里的路径是 Linux 路径（`/database/home/...`），本地运行前先改成你自己的路径。

### 5.1 训练（step2）

改两处路径后直接运行，或在 `__main__` 里调用：

```python
# step2_automodel_ph.py 内
# ① load_data() 里：excel_path 改成标签文件路径
excel_path = r'D:\...\你的标签文件.xlsx'

# ② main_ph_workflow() 里：savepath 改成模型保存目录
savepath = r'D:\PycharmProjects\pythonProject8\Ph\model_label1'

# ③ 训练单个/多个标签
main_ph_workflow(mode='all', labels_to_train=['label1'])
# main_ph_workflow(mode='all', labels_to_train=['label1','label2','label3'])
```

### 5.2 外部验证（step4）

```python
evaluate_external_validation(
    model_dir=r'D:\PycharmProjects\pythonProject8\Ph\model_label1',   # 训练时 savepath
    external_file=r'D:\...\外部验证集.xlsx',
    label_name='label1',
)
```

### 5.3 依赖

- `autogluon.tabular`（AutoGluon 表格预测）
- `xgboost`、`lightgbm`、`catboost`、`scikit-learn`
- `pandas`、`numpy`、`matplotlib`、`seaborn`
- 依赖包 `pymagic.EcgClassify`（提供 `load_ecg_feature`，即「加载特征 X」的实现，不在此文件夹内，需在 `sys.path` 上可用）

---

## 6. 输出结果

**训练（`classify_autogluon` 写入 `savepath/`）：**

| 文件 | 内容 |
|---|---|
| （模型目录本身） | AutoGluon 训练好的模型 |
| `leaderboard.csv` | 各模型得分排行榜 |
| `metrics_report.csv` / `metrics_summary.csv` | 分类报告 / 汇总指标（accuracy、AUC、NPV 等） |
| `confusion_matrix.csv` | 混淆矩阵 |
| `feature_importance.csv` | 特征重要性 |
| `y_true_score.csv` | 预测概率 + 真实/预测标签 |

**验证（`step4` 写入 `{model_dir}_external_results/`）：**

| 文件 | 内容 |
|---|---|
| `classification_report.csv` | 分类报告 |
| `confusion_matrix.csv` | 混淆矩阵 |
| `predictions.csv` | 每条样本的 y_true / y_pred / 各类概率 |
| `external_samples_used_new.csv` | 实际进入模型的外部样本 |

---

## 7. 注意事项

1. **防数据泄漏**：切分用 `GroupShuffleSplit`（按 `subject_id` 分组），同一受试者的多条记录不会被同时放进训练集和测试集。
2. **标签口径**：只有「收缩」压参与分级；「平均/平均压」会被过滤，非收缩或数值缺失归为正常（0）。
3. **`db` 字段**：`create_labels` 会把 `db` 设为 `'qinghaiai'`，`load_ecg_feature` 据此定位对应的特征表格；外部验证集若无 `db` 列会被默认设为 `'qinghaiai2'`。
