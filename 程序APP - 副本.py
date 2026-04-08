import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 页面配置
st.set_page_config(page_title="青少年SZ/PD鉴别诊断", layout="centered")

# 加载模型和归一化器（确保两个文件在同一目录下）
model = joblib.load('best_catboost_model.pkl')
scaler = joblib.load('scaler.pkl')          # 必须与训练时保存的 scaler 一致

# 特征名称（顺序需与训练时完全一致）
feature_names = ["NEU%", "EO%", "PDW", "PRL", "LDH", "FT4"]

# 自定义CSS样式（保持美观）
st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .input-header {
        font-size: 24px;
        font-weight: 600;
        color: #34495e;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .result-text {
        font-size: 26px;
        font-weight: bold;
        text-align: center;
    }
    .probability {
        font-size: 40px;
        font-weight: 800;
        color: #0066cc;
    }
    div.stButton > button {
        font-size: 18px;
        padding: 0.5em 2em;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 主标题
st.markdown('<div class="main-title">基于CatBoost模型鉴别诊断青少年SZ及PD</div>', unsafe_allow_html=True)

# 输入区域
st.markdown('<div class="input-header">📝 输入以下特征检测值：</div>', unsafe_allow_html=True)

feature_values = []
cols = st.columns(2)
for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        value = st.number_input(
            label=f"{feature}",
            value=0.0,
            step=0.1,
            format="%.4f",
            key=feature,
        )
        feature_values.append(value)

# 预测按钮
if st.button("🚀 开始预测", use_container_width=True):
    # 1. 构建原始输入DataFrame
    input_df = pd.DataFrame([feature_values], columns=feature_names)

    # 2. 使用训练时保存的scaler进行归一化（关键步骤）
    input_scaled = scaler.transform(input_df)

    # 3. 模型预测（使用归一化后的数据）
    predicted_class = model.predict(input_scaled)[0]
    predicted_proba = model.predict_proba(input_scaled)[0]

    # 4. 根据预测类别动态设置显示内容
    if predicted_class == 0:
        class_name = "SZ"
        prob = predicted_proba[0] * 100   # 类别0的概率
        other_class = "PD"
        other_prob = predicted_proba[1] * 100
    else:
        class_name = "PD"
        prob = predicted_proba[1] * 100   # 类别1的概率
        other_class = "SZ"
        other_prob = predicted_proba[0] * 100

    # 5. 显示预测结果（动态文字）
    st.markdown("---")
    st.markdown('<div class="result-text">预测结果</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='background-color:#f0f6ff; padding:20px; border-radius:15px; text-align:center; margin-top:10px;'>
            <p style='font-size:22px; font-weight:bold; margin-bottom:10px;'>基于特征值预测为 
            <span style='color:#ff4b4b;'>{class_name}</span> 的可能性是</p>
            <p class="probability">{prob:.2f}%</p>
            <p style='margin-top:15px; color:#555;'>(注：标签0 = SZ，标签1 = PD)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 6. 展示完整概率信息（可选）
    with st.expander("🔍 查看详细预测概率"):
        prob_df = pd.DataFrame({
            '疾病类别': ['SZ (标签0)', 'PD (标签1)'],
            '预测概率 (%)': [f"{predicted_proba[0]*100:.2f}%", f"{predicted_proba[1]*100:.2f}%"]
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

    # 7. 显示输入的原始特征值（供核对）
    with st.expander("📋 查看输入特征值（原始数值）"):
        st.dataframe(input_df, use_container_width=True)
