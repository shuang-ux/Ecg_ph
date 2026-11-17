
# %%
#----------------------------------------心电+年龄性别-------------------------------
import sys
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.common.utils.resource_utils import ResourceManager
import xgboost as xgb
import re

# 设置路径
sys.path.append("../")
sys.path.append("./")

# 清除并重新导入模块
module_name = 'pymagic.EcgClassify.classify_autogluon_bin'
if module_name in sys.modules:
    del sys.modules[module_name]

from pymagic.EcgClassify.classify_autogluon_bin import classify_autogluon

# 1. 数据加载函数
def load_data():
    # """加载数据，可选择从pickle或Excel加载"""
    # # 尝试从pickle加载
    # pickle_path = "data_V8.pkl"
    # if os.path.exists(pickle_path):
    #     with open(pickle_path, "rb") as f:
    #         df = pickle.load(f)
    #     print(f"从pickle文件加载数据，共{len(df)}条记录")
    # else:
        # 从Excel加载
    excel_path = '/database/home/wushuang/pywave/apptrpe/qinghaiai_ecg_echo_v10.1.xlsx'
    df = pd.read_excel(excel_path, dtype={'subject_id': str})
    print(f"从Excel文件加载数据，共{len(df)}条记录")
    return df

# 2. 肺动脉高压分级函数
def label_ph(pressure_type, pressure_value, mode='all'):
    """
    肺动脉高压分级：
    0: 正常
    1: 轻度 (35-50 mmHg)
    2: 中度 (51-70 mmHg)
    3: 重度 (>70 mmHg)
    
    Parameters:
    -----------
    mode : str
        'all': 包含轻度级别
        'exclude_mid_range': 排除35-50之间的数据
    """
    if pd.isna(pressure_type) or pd.isna(pressure_value):
        return 0
    
    if not isinstance(pressure_type, str):
        return 0
    
    if pressure_type != "收缩":
        return 0
    
    try:
        pressure_value = float(pressure_value)
    except (ValueError, TypeError):
        return 0
    
    # 根据模式处理
    if mode == 'exclude_mid_range':
        # 排除35-50范围的数据
        if 35 <= pressure_value <= 50:
            return -1  # 标记为要排除的数据
    
    # 分级逻辑
    if pressure_value < 35:
        return 0
    elif 35 <= pressure_value <= 50:
        return 1
    elif 51 <= pressure_value <= 70:
        return 2
    else:  # > 70
        return 3


# 3. 创建不同标签的函数
def create_labels(df, mode='all'):
    """
    创建不同的标签
    
    Parameters:
    -----------
    mode : str
        'all': 包含轻度级别
        'exclude_mid_range': 排除35-50之间的数据
    """
    df_processed = df.copy()
    
    # 筛选肺动脉高压数据集
    df_processed = df_processed[~df_processed['肺动脉压类型'].isin(['平均', '平均压'])].copy()
    
    # 应用肺动脉高压分级
    df_processed['ph_level'] = df_processed.apply(
        lambda row: label_ph(row['肺动脉压类型'], row['肺动脉压数值'], mode), axis=1
    )
    df_processed['db'] = 'qinghaiai'

    # 根据模式处理
    if mode == 'exclude_mid_range':
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed['ph_level'] != -1]  # 删除标记为-1的行
        deleted_count = initial_count - len(df_processed)
        print(f"\n排除35-50mmHg范围数据: {deleted_count} 条")
        print(f"处理后肺动脉高压分布:")
        print(df_processed['ph_level'].value_counts().sort_index())
        
        # 只创建label1
        df_processed['label1'] = (df_processed['ph_level'] > 0).astype(int)
        return df_processed, ['label1']
    
    else:  # mode == 'all'
        # 创建多种标签
        # label1: PH 中度及以上
        df_processed['label1'] = (df_processed['ph_level'] > 1).astype(int)
        # label2: PH 轻度及以上
        df_processed['label2'] = (df_processed['ph_level'] > 0).astype(int)
        # label3: 多分类，中度/重度合并为2
        df_processed['label3'] = np.where(df_processed['ph_level'] > 1, 2, df_processed['ph_level'])
        
        print("\n各标签分布:")
        for label in ['label1', 'label2', 'label3']:
            print(f"{label}: {df_processed[label].value_counts().sort_index().to_dict()}")
        
        return df_processed, ['label1', 'label2', 'label3']

# 6. 主工作流程函数
def main_ph_workflow(mode='all', labels_to_train=None, check_gpu=True):
    """
    肺动脉高压分类主工作流程
    
    Parameters:
    -----------
    mode : str
        'all': 包含轻度级别
        'exclude_mid_range': 排除35-50之间的数据
    labels_to_train : list
        要训练的标签列表
    check_gpu : bool
        是否检查GPU
    """
    # 加载数据
    df = load_data()
    print(f"数据列名: {list(df.columns)}")
    print(f"数据大小: {len(df)}")
    
    # 创建标签
    df_processed, default_labels = create_labels(df, mode)
    
    # 确定要训练的标签
    if labels_to_train is None:
        labels_to_train = default_labels
    
    # 打印系统信息
    if check_gpu:
        print("\n系统信息:")
        print(f"GPU数量: {ResourceManager.get_gpu_count()}")
        print(f"XGBoost版本: {xgb.__version__}")
    
    # 模型训练
    for label in labels_to_train:
        print(f"\n{'='*60}")
        print(f"训练标签: {label}")
        print(f"类别分布: {dict(df_processed[label].value_counts().sort_index())}")
        

            # 准备数据
        df_classify = df_processed[['record', 'subject_id', 'db', label,'age','gender']].copy()
        
        # 设置超参数
        hyperparameters = {'GBM': {}, 'CAT': {}, 'RF': {}, 'XGB': {}}
        
        # 设置保存路径
        if mode == 'exclude_mid_range':
            savepath = f'/database/home/wushuang/pywave/apptrpe_newdata/mid2(PH二分类排轻)/{label}'
        else:
            savepath = f'/database/home/wushuang/pywave/apptrpe/onlyEcg/Phf0'
        
        # 根据标签类型选择训练参数
        if label == 'label3':  # 三分类
            classify_autogluon(
                df_classify,
                savepath=savepath,
                label_column=label,
                eval_metric='f1',
                hyperparameters=hyperparameters
            )
        else:  # 二分类
            classify_autogluon(
                df_classify,
                savepath=savepath,
                label_column=label,
                eval_metric='f1',
                hyperparameters=hyperparameters
            )
        
        print(f"模型已保存到: {savepath}")

# 8. 执行主程序
if __name__ == "__main__":
    # 版本1：包含轻度级别
    print("="*80)
    print("版本1: 包含轻度肺动脉高压数据")
    print("="*80)
    main_ph_workflow(mode='all', labels_to_train=['label1'])
    
    # # 版本2：排除35-50范围数据
    # print("\n" + "="*80)
    # print("版本2: 排除35-50mmHg范围的肺动脉高压数据")
    # print("="*80)
    # main_ph_workflow(mode='exclude_mid_range', labels_to_train=['label1'])


# %%
#----------------------------------------不含年龄性别---------------------------------
import sys
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.common.utils.resource_utils import ResourceManager
import xgboost as xgb
import re

# 设置路径
sys.path.append("../")
sys.path.append("./")

# 清除并重新导入模块
module_name = 'pymagic.EcgClassify.classify_autogluon_bin'
if module_name in sys.modules:
    del sys.modules[module_name]

from pymagic.EcgClassify.classify_autogluon_bin import classify_autogluon

# 1. 数据加载函数
def load_data():
    # """加载数据，可选择从pickle或Excel加载"""
    # # 尝试从pickle加载
    # pickle_path = "data_V8.pkl"
    # if os.path.exists(pickle_path):
    #     with open(pickle_path, "rb") as f:
    #         df = pickle.load(f)
    #     print(f"从pickle文件加载数据，共{len(df)}条记录")
    # else:
        # 从Excel加载
    excel_path = '/database/home/wushuang/pywave/apptrpe/qinghaiai_ecg_echo_v10.1.xlsx'
    df = pd.read_excel(excel_path, dtype={'subject_id': str})
    print(f"从Excel文件加载数据，共{len(df)}条记录")
    return df

# 2. 肺动脉高压分级函数
def label_ph(pressure_type, pressure_value, mode='all'):
    """
    肺动脉高压分级：
    0: 正常
    1: 轻度 (35-50 mmHg)
    2: 中度 (51-70 mmHg)
    3: 重度 (>70 mmHg)
    
    Parameters:
    -----------
    mode : str
        'all': 包含轻度级别
        'exclude_mid_range': 排除35-50之间的数据
    """
    if pd.isna(pressure_type) or pd.isna(pressure_value):
        return 0
    
    if not isinstance(pressure_type, str):
        return 0
    
    if pressure_type != "收缩":
        return 0
    
    try:
        pressure_value = float(pressure_value)
    except (ValueError, TypeError):
        return 0
    
    # 根据模式处理
    if mode == 'exclude_mid_range':
        # 排除35-50范围的数据
        if 35 <= pressure_value <= 50:
            return -1  # 标记为要排除的数据
    
    # 分级逻辑
    if pressure_value < 35:
        return 0
    elif 35 <= pressure_value <= 50:
        return 1
    elif 51 <= pressure_value <= 70:
        return 2
    else:  # > 70
        return 3


# 3. 创建不同标签的函数
def create_labels(df, mode='all'):
    """
    创建不同的标签
    
    Parameters:
    -----------
    mode : str
        'all': 包含轻度级别
        'exclude_mid_range': 排除35-50之间的数据
    """
    df_processed = df.copy()
    
    # 筛选肺动脉高压数据集
    df_processed = df_processed[~df_processed['肺动脉压类型'].isin(['平均', '平均压'])].copy()
    
    # 应用肺动脉高压分级
    df_processed['ph_level'] = df_processed.apply(
        lambda row: label_ph(row['肺动脉压类型'], row['肺动脉压数值'], mode), axis=1
    )
    df_processed['db'] = 'qinghaiai'

    # 根据模式处理
    if mode == 'exclude_mid_range':
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed['ph_level'] != -1]  # 删除标记为-1的行
        deleted_count = initial_count - len(df_processed)
        print(f"\n排除35-50mmHg范围数据: {deleted_count} 条")
        print(f"处理后肺动脉高压分布:")
        print(df_processed['ph_level'].value_counts().sort_index())
        
        # 只创建label1
        df_processed['label1'] = (df_processed['ph_level'] > 0).astype(int)
        return df_processed, ['label1']
    
    else:  # mode == 'all'
        # 创建多种标签
        # label1: PH 中度及以上
        df_processed['label1'] = (df_processed['ph_level'] > 1).astype(int)
        # label2: PH 轻度及以上
        df_processed['label2'] = (df_processed['ph_level'] > 0).astype(int)
        # label3: 多分类，中度/重度合并为2
        df_processed['label3'] = np.where(df_processed['ph_level'] > 1, 2, df_processed['ph_level'])
        
        print("\n各标签分布:")
        for label in ['label1', 'label2', 'label3']:
            print(f"{label}: {df_processed[label].value_counts().sort_index().to_dict()}")
        
        return df_processed, ['label1', 'label2', 'label3']

# 6. 主工作流程函数
def main_ph_workflow(mode='all', labels_to_train=None, check_gpu=True):
    """
    肺动脉高压分类主工作流程
    
    Parameters:
    -----------
    mode : str
        'all': 包含轻度级别
        'exclude_mid_range': 排除35-50之间的数据
    labels_to_train : list
        要训练的标签列表
    check_gpu : bool
        是否检查GPU
    """
    # 加载数据
    df = load_data()
    print(f"数据列名: {list(df.columns)}")
    print(f"数据大小: {len(df)}")
    
    # 创建标签
    df_processed, default_labels = create_labels(df, mode)
    
    # 确定要训练的标签
    if labels_to_train is None:
        labels_to_train = default_labels
    
    # 打印系统信息
    if check_gpu:
        print("\n系统信息:")
        print(f"GPU数量: {ResourceManager.get_gpu_count()}")
        print(f"XGBoost版本: {xgb.__version__}")
    
    # 模型训练
    for label in labels_to_train:
        print(f"\n{'='*60}")
        print(f"训练标签: {label}")
        print(f"类别分布: {dict(df_processed[label].value_counts().sort_index())}")
        

            # 准备数据
        df_classify = df_processed[['record', 'subject_id', 'db', label]].copy()
        
        # 设置超参数
        hyperparameters = {'GBM': {}, 'CAT': {}, 'RF': {}, 'XGB': {}}
        
        # 设置保存路径
        if mode == 'exclude_mid_range':
            savepath = f'/database/home/wushuang/pywave/apptrpe_newdata/mid2(PH二分类排轻)/{label}'
        else:
            savepath = f'/database/home/wushuang/pywave/apptrpe/Ph_new_onlyEcg'
        
        # 根据标签类型选择训练参数
        if label == 'label3':  # 三分类
            classify_autogluon(
                df_classify,
                savepath=savepath,
                label_column=label,
                eval_metric='f1',
                hyperparameters=hyperparameters
            )
        else:  # 二分类
            classify_autogluon(
                df_classify,
                savepath=savepath,
                label_column=label,
                eval_metric='f1',
                hyperparameters=hyperparameters
            )
        
        print(f"模型已保存到: {savepath}")

# 8. 执行主程序
if __name__ == "__main__":
    # 版本1：包含轻度级别
    print("="*80)
    print("版本1: 包含轻度肺动脉高压数据")
    print("="*80)
    main_ph_workflow(mode='all', labels_to_train=['label1'])
    
    # # 版本2：排除35-50范围数据
    # print("\n" + "="*80)
    # print("版本2: 排除35-50mmHg范围的肺动脉高压数据")
    # print("="*80)
    # main_ph_workflow(mode='exclude_mid_range', labels_to_train=['label1'])


# %%
