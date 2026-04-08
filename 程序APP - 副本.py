import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 页面配置（可选，增加整体宽度）
st.set_page_config(page_title="青少年SZ/PD鉴别诊断", layout="centered")

# 加载模型（确保 catboost_model.pkl 在当前目录）
model = joblib.load('catboost_model.pkl')

# 特征名称（顺序需与训练时一致）
feature_names = ["FT4", "PRL", "PDW", "NEU%", "LDH", "EO%"]

# 自定义CSS，美化整体字体和按钮
st.markdown(
    """
    <style>
    /* 主标题 */
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    /* 输入区域标题 */
    .input-header {
        font-size: 24px;
        font-weight: 600;
        color: #34495e;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    /* 预测结果卡片中的主要文字 */
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
    /* 按钮样式（增大内边距和字体） */
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

# 美化后的主标题
st.markdown('<div class="main-title">基于CatBoost模型鉴别诊断青少年SZ及PD</div>', unsafe_allow_html=True)

# 输入区域标题
st.markdown('<div class="input-header">📝 输入以下特征检测值：</div>', unsafe_allow_html=True)

# 动态生成输入项（保持原数值输入方式）
feature_values = []
cols = st.columns(2)  # 分成两列布局，更紧凑美观
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
    # 转换为 DataFrame
    input_df = pd.DataFrame([feature_values], columns=feature_names)

    # 模型预测
    predicted_class = model.predict(input_df)[0]
    predicted_proba = model.predict_proba(input_df)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果（美化卡片）
    st.markdown("---")
    st.markdown('<div class="result-text">📊 预测结果</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='background-color:#f0f6ff; padding:20px; border-radius:15px; text-align:center; margin-top:10px;'>
            <p style='font-size:22px; font-weight:bold; margin-bottom:10px;'>基于特征值预测为 
            <span style='color:#ff4b4b;'>PD</span> 的可能性是</p>
            <p class="probability">{probability:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 可选：展开显示输入特征值
    with st.expander("🔍 查看输入特征值"):
        st.dataframe(pd.DataFrame([feature_values], columns=feature_names), use_container_width=True)
