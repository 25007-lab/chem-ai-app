import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

# 1. 저장해둔 모델 불러오기
model = joblib.load('solubility_model.pkl')

st.title("🧪 화학 분자 다차원 분석 AI 웹사이트")
st.write("분자 구조식(SMILES)을 입력하면 AI 용해도 예측과 함께 다양한 분자적 성질을 분석해 줍니다.")

# 2. 사용자 입력창
smiles_input = st.text_input("분자식(SMILES)을 입력하세요 (예: CCO, c1ccccc1 등):", "CCO")

if st.button("분석 및 예측하기"):
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # 탭을 활용해 결과 깔끔하게 나누기
            tab1, tab2 = st.tabs(["📊 AI 용해도 예측", "🔬 분자 구조 및 극성 분석"])
            
            with tab1:
                # 예측을 위한 데이터 변환 및 예측
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                features = np.array(fp).reshape(1, -1)
                prediction = model.predict(features)
                
                st.subheader("용해도(LogS) 예측 결과")
                st.success(f"이 분자의 예측 용해도 값(LogS)은 약 **{prediction[0]:.2f}**입니다.")
                st.caption("※ LogS 값이 클수록(0에 가까울수록) 물에 잘 녹는 경향이 있음을 의미합니다.")
            
            with tab2:
                st.subheader("분자 구조 및 물리화학적 특징 분석")
                
                # 분자량 계산
                mw = Descriptors.MolWt(mol)
                # 극성 표면적(TPSA) 계산 -> 극성 여부를 파악하는 좋은 지표
                tpsa = Descriptors.TPSA(mol)
                # 옥탄올-물 분배계수 (LogP) -> 극성/비극성 성향 판단
                logp = Descriptors.MolLogP(mol)
                
                st.metric(label="분자량 (Molecular Weight)", value=f"{mw:.2f} g/mol")
                st.metric(label="극성 표면적 (TPSA)", value=f"{tpsa:.2f} Å²", 
                          help="값이 클수록 분자의 극성이 크고 수소 결합에 유리함을 나타냅니다.")
                st.metric(label="분배계수 (LogP)", value=f"{logp:.2f}", 
                          help="양수면 지용성(비극성 성향), 음수면 수용성(극성 성향)이 강합니다.")
                
                # 간단한 극성 추정 가이드
                if tpsa > 60 or logp < 0:
                    st.info("💡 **분자 성향 분석**: 이 분자는 극성 작용기를 포함하고 있어 상대적으로 **극성** 성향을 띨 가능성이 높습니다.")
                else:
                    st.info("💡 **분자 성향 분석**: 이 분자는 탄화수소 사슬 등이 우세하여 상대적으로 **비극성** 성향을 띨 가능성이 높습니다.")
                    
        else:
            st.error("올바른 SMILES 형식이 아닙니다. 올바른 구조식을 입력해주세요.")

st.info("💡 힌트: 물(O), 에탄올(CCO), 아세트산(CC(=O)O), 카페인(Cn1cnc2c1c(=O)n(c(=O)n2C)C) 등을 입력해 비교해 보세요!")
