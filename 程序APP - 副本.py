import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载模型（支持 CalibratedClassifierCV 或任何 sklearn 兼容模型）
model = joblib.load('catboost_model.pkl')

# 定义特征名称（顺序需与训练时一致）
feature_names = ["FT4", "PRL", "PDW", "NEU%", "LDH", "EO%"]

# 缩小标题字体（使用 HTML 的 h4 标签）
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

# 转换为 DataFrame（便于模型预测）
input_df = pd.DataFrame([feature_values], columns=feature_names)

# 预测按钮
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(input_df)[0]
    predicted_proba = model.predict_proba(input_df)[0]
    probability = predicted_proba[predicted_class] * 100

    # 直接使用 st.markdown 显示结果（避免 matplotlib 字体问题）
    st.markdown(f"<h3>📊 预测结果</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background-color:#f0f2f6; padding:10px; border-radius:10px'>"
        f"<p style='font-size:20px; font-weight:bold;'>基于特征值预测为 <span style='color:#ff4b4b'>PD</span> 的可能性是 <span style='color:#0066cc'>{probability:.2f}%</span></p>"
        f"</div>",
        unsafe_allow_html=True
    )

    # 可选：显示特征输入值（调试用）
    with st.expander("查看输入特征值"):
        st.dataframe(pd.DataFrame([feature_values], columns=feature_names))
