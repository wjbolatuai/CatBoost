import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的CatBoost模型
model = joblib.load('catboost_model.pkl')

# 定义特征名称和类型（仅用于生成输入框，不再包含范围限制）
feature_names = ["FT4", "PRL", "PDW", "NEU%", "LDH", "EO%"]

# 缩小标题字体（原st.title改为markdown + h3）
st.markdown("<h3>基于CatBoost模型鉴别诊断青少年SZ及PD</h3>", unsafe_allow_html=True)

# 动态生成输入项（无数值范围限制）
st.header("输入以下特征检测值:")
feature_values = []
for feature in feature_names:
    # 使用普通的数字输入框，不限制min/max，默认值0.0
    value = st.number_input(
        label=f"{feature}",
        value=0.0,
        step=0.1,
        format="%.4f"
    )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与SHAP可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用Matplotlib渲染指定字体
    text = f"基于特征值预测为PD的可能性是 {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 生成SHAP力图（适用于二分类或多分类）
    class_index = predicted_class
    # 注意：shap_values可能是列表（多分类）或数组（二分类）
    if isinstance(shap_values, list):
        shap_fig = shap.force_plot(
            explainer.expected_value[class_index],
            shap_values[class_index],
            pd.DataFrame([feature_values], columns=feature_names),
            matplotlib=True,
        )
    else:
        shap_fig = shap.force_plot(
            explainer.expected_value,
            shap_values,
            pd.DataFrame([feature_values], columns=feature_names),
            matplotlib=True,
        )
    # 保存并显示SHAP图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")