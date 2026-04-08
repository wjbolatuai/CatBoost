import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings

# 设置 matplotlib 中文字体（避免方框乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False                   # 用来正常显示负号

# 加载保存的模型（可能是 CalibratedClassifierCV 包装的 CatBoost）
model = joblib.load('catboost_model.pkl')

# 如果模型是 CalibratedClassifierCV，尝试提取内部的 CatBoost 模型供 SHAP 使用
if hasattr(model, 'calibrated_classifiers_'):
    # 假设第一个校准器的基础模型就是原始 CatBoost
    base_model = model.calibrated_classifiers_[0].base_estimator
    st.info("检测到模型为 CalibratedClassifierCV，已自动提取内部 CatBoost 模型用于 SHAP 解释。")
else:
    base_model = model

# 定义特征名称（顺序需与训练时一致）
feature_names = ["FT4", "PRL", "PDW", "NEU%", "LDH", "EO%"]

# 缩小标题字体
st.markdown("<h4>基于CatBoost模型鉴别诊断青少年SZ及PD</h4>", unsafe_allow_html=True)

# 动态生成输入项（无任何数值范围限制）
st.header("输入以下特征检测值:")
feature_values = []
for feature in feature_names:
    value = st.number_input(
        label=f"{feature}",
        value=0.0,
        step=0.1,
        format="%.4f"
    )
    feature_values.append(value)

# 转换为 DataFrame 格式（便于 SHAP 使用）
input_df = pd.DataFrame([feature_values], columns=feature_names)

# 预测按钮
if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(input_df)[0]
    predicted_proba = model.predict_proba(input_df)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果（使用 matplotlib 渲染，避免字体问题）
    text = f"基于特征值预测为PD的可能性是 {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center',
            transform=ax.transAxes)
    ax.axis('off')
    st.pyplot(fig)   # 直接显示，不保存临时文件
    plt.close(fig)

    # ------------------- SHAP 解释（仅当 base_model 可用时） -------------------
    try:
        # 使用 TreeExplainer 解释内部 CatBoost 模型
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(input_df)

        # 判断是多分类还是二分类
        if isinstance(shap_values, list):
            # 多分类：shap_values 是一个列表，每个元素是 [样本数, 特征数]
            class_idx = predicted_class
            shap_vals_class = shap_values[class_idx][0]   # 取第一个样本的 SHAP 值
            expected_value = explainer.expected_value[class_idx]
        else:
            # 二分类：shap_values 是二维数组 [样本数, 特征数]
            shap_vals_class = shap_values[0]
            expected_value = explainer.expected_value

        # 生成 SHAP 力图（force plot）并显示
        shap_fig = shap.force_plot(
            expected_value,
            shap_vals_class,
            input_df.iloc[0, :],
            matplotlib=True,
            show=False
        )
        st.pyplot(shap_fig)
        plt.close(shap_fig)

    except Exception as e:
        st.warning(f"SHAP 解释图生成失败（模型类型可能不兼容）：{e}")
