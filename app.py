import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. 저장해둔 모델 불러오기
model = joblib.load('solubility_model.pkl')

st.title("🧪 화학 분자 물성 예측 AI")
st.write("분자 구조식(SMILES)을 입력하면 AI가 용해도를 예측해 줍니다.")

# 2. 사용자 입력창
smiles_input = st.text_input("분자식(SMILES)을 입력하세요 (예: CCO, c1ccccc1 등):")

if st.button("예측하기"):
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # 시각화
            from rdkit.Chem.Draw import rdMolDraw2D

            # 최신 RDKit에 맞춘 안정적인 2차원 구조 이미지 생성 코드
            drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            png_data = drawer.GetDrawingText()

            st.image(png_data, caption="입력하신 분자 구조")
            
            # 예측을 위한 데이터 변환
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            features = np.array(fp).reshape(1, -1)
            
            # 예측 실행
            prediction = model.predict(features)
            st.success(f"이 분자의 예측 용해도 값(LogS)은 약 {prediction[0]:.2f}입니다.")
        else:
            st.error("올바른 SMILES 형식이 아닙니다.")

st.info("💡 힌트: 카페인(Cn1cnc2c1c(=O)n(c(=O)n2C)C), 아스피린(CC(=O)Oc1ccccc1C(=O)O) 등을 입력해 보세요!")
