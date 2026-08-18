import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski

st.set_page_config(page_title="화학 분자 분석 AI", layout="wide")

st.title("🧪 화학 분자 다차원 분석 및 ESOL 예측 시스템")
st.write("분자 구조식(SMILES)을 입력하면 순수 ESOL 수학 모델과 분자 디스크립터를 실시간으로 통합 산출합니다.")

# 사용자 입력창
smiles_input = st.text_input("분자식(SMILES)을 입력하세요 (예: CCO, CC(=O)Oc1ccccc1C(=O)O 등):", "CCO")

# ESOL 모델 및 분자 특성 계산 함수 (LogS, LogP, MW, RB, AP, TPSA 통합 계산)
def calculate_esol_and_features(mol):
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms > 0:
        aromatic_heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        ap = aromatic_heavy_atoms / heavy_atoms
    else:
        ap = 0.0
        
    # Delaney의 ESOL 공식
    log_s = 0.16 - (0.63 * logp) - (0.0062 * mw) + (0.066 * rb) - (0.74 * ap)
    return log_s, logp, mw, rb, ap, tpsa

if st.button("분석 및 예측하기"):
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # --- 화면 구성 (좌우 분할) ---
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🔍 입력된 분자 구조")
                img_url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles_input}/image"
                st.image(img_url, caption="분자 구조 이미지", width=350)

            with col2:
                st.subheader("📊 통합 분자 성질 및 용해도 분석")
                
                # 모든 지표 한 번에 계산
                log_s, logp, mw, rb, ap, tpsa = calculate_esol_and_features(mol)
                
                # 용해도 결과 강조 출력
                st.success(f"**예측된 용해도 (LogS): {log_s:.2f}**")
                
                # 상세 물리화학적 지표들을 메트릭으로 한눈에 배치 (RB와 AP 포함)
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="분자량 (MW)", value=f"{mw:.2f} g/mol")
                    st.metric(label="회전 결합 수 (RB)", value=f"{rb}")
                with m_col2:
                    st.metric(label="분배계수 (LogP)", value=f"{logp:.2f}")
                    st.metric(label="방향족 비율 (AP)", value=f"{ap:.2f}")
                with m_col3:
                    st.metric(label="극성 표면적 (TPSA)", value=f"{tpsa:.2f} Å²")
                    
                # 극성 성향 분석 가이드
                if tpsa > 60 or logp < 0:
                    st.info("💡 **분자 성향 분석**: 이 분자는 극성 작용기를 포함하고 있어 상대적으로 **극성** 성향을 띨 가능성이 높습니다.")
                else:
                    st.info("💡 **분자 성향 분석**: 이 분자는 탄화수소 사슬 등이 우세하여 상대적으로 **비극성** 성향을 띨 가능성이 높습니다.")
                        
        else:
            st.error("올바른 SMILES 형식이 아닙니다. 올바른 구조식을 입력해주세요.")

st.markdown("---")
st.info("💡 **팁**: 에탄올(`CCO`), 아스피린(`CC(=O)Oc1ccccc1C(=O)O`), 아세트산(`CC(=O)O`) 같은 유기 화합물들을 입력하면 ESOL 알고리즘이 아주 정확하게 작동합니다.")
