import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski

st.set_page_config(page_title="화학 분자 분석 AI", layout="wide")

st.title("🧪 화학 분자 다차원 분석 및 용해도 예측 시스템")
st.write("분자 구조식(SMILES)을 입력하면 ESOL 알고리즘을 통한 정확한 용해도(LogS) 계산과 구조 분석을 제공합니다.")

# 2. 사용자 입력창
smiles_input = st.text_input("분자식(SMILES)을 입력하세요 (예: O=C=O, CCO 등):", "O=C=O")

# ESOL 모델을 이용한 객관적인 용해도(LogS) 계산 함수
def calculate_esol(mol):
    # 1. 분배계수 (LogP)
    logp = Descriptors.MolLogP(mol)
    # 2. 분자량 (MW)
    mw = Descriptors.MolWt(mol)
    # 3. 회전 가능한 결합 수 (Rotatable Bonds)
    rb = Lipinski.NumRotatableBonds(mol)
    
    # 4. 방향족 비율 (Aromatic Proportion) = 방향족 중원자 수 / 전체 중원자 수
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms > 0:
        aromatic_heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        ap = aromatic_heavy_atoms / heavy_atoms
    else:
        ap = 0.0
        
    # Delaney의 ESOL 공식 적용
    log_s = 0.16 - (0.63 * logp) - (0.0062 * mw) + (0.066 * rb) - (0.74 * ap)
    return log_s, logp, mw

if st.button("분석 및 예측하기"):
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # --- 화면 구성 ---
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🔍 입력된 분자 구조")
                img_url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles_input}/image"
                st.image(img_url, caption="분자 구조 이미지", width=350)

            with col2:
                tab1, tab2 = st.tabs(["📊 용해도(LogS) 분석", "🔬 물리화학적 상세 분석"])
                
                with tab1:
                    st.subheader("ESOL 알고리즘 기반 용해도 예측")
                    
                    # 알고리즘으로 직접 계산
                    log_s, logp, mw = calculate_esol(mol)
                    
                    st.success(f"이 분자의 계산된 용해도 값(LogS)은 약 **{log_s:.2f}**입니다.")
                    st.caption("※ LogS 값이 클수록(0에 가까울수록) 물에 잘 녹는 경향이 있음을 의미합니다.")
                    
                with tab2:
                    st.subheader("분자 구조 특징 분석")
                    
                    tpsa = Descriptors.TPSA(mol)
                    
                    st.metric(label="분자량 (Molecular Weight)", value=f"{mw:.2f} g/mol")
                    st.metric(label="극성 표면적 (TPSA)", value=f"{tpsa:.2f} Å²")
                    st.metric(label="분배계수 (LogP)", value=f"{logp:.2f}")
                    
                    if tpsa > 60 or logp < 0:
                        st.info("💡 **분자 성향 분석**: 이 분자는 극성 작용기를 포함하고 있어 상대적으로 **극성** 성향을 띨 가능성이 높습니다.")
                    else:
                        st.info("💡 **분자 성향 분석**: 이 분자는 탄화수소 사슬 등이 우세하여 상대적으로 **비극성** 성향을 띨 가능성이 높습니다.")
                        
        else:
            st.error("올바른 SMILES 형식이 아닙니다. 올바른 구조식을 입력해주세요.")

st.markdown("---")
st.info("💡 **팁**: 이산화탄소(`O=C=O`), 에탄올(`CCO`), 아스피린(`CC(=O)Oc1ccccc1C(=O)O`) 등을 입력해 테스트해 보세요.")
