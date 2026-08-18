import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

# 1. 저장해둔 모델 불러오기
model = joblib.load('solubility_model.pkl')

st.set_page_config(page_title="화학 분자 분석 AI", layout="wide")

st.title("🧪 화학 분자 다차원 분석 AI 웹사이트")
st.write("분자 구조식(SMILES)을 입력하면 AI 용해도 예측과 함께 분자 구조 이미지 및 다양한 성질을 분석해 줍니다.")

# 2. 사용자 입력창
smiles_input = st.text_input("분자식(SMILES)을 입력하세요 (예: CCO, c1ccccc1 등):", "CCO")

# 3. 분자 구조를 그려주는 함수 (SVG 방식 - 서버 환경에서 가장 안정적)
def render_mol_svg(mol):
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    # 분자 구조에 하이라이트나 색상을 입히는 옵션 (선택 사항)
    # 옵션 추가를 원하면 주석 해제: rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # Streamlit에서 SVG를 표시하기 위해 약간의 조치가 필요
    return svg.replace('svg:', '')

if st.button("분석 및 예측하기"):
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # --- 화면 구성 ---
            col1, col2 = st.columns([1, 1]) # 화면을 좌우로 나눔
            
            with col1:
                st.subheader("🔍 입력된 분자 구조")
                try:
                    # 위에서 만든 함수로 SVG 이미지 생성 및 출력
                    mol_svg = render_mol_svg(mol)
                    st.image(mol_svg, output_format="SVG", width=400)
                except Exception as e:
                    st.error(f"이미지를 그리는 중에 오류가 발생했습니다: {e}")
                    st.warning("이미지는 안 나오지만 분석 기능은 계속 이용 가능합니다.")

            with col2:
                # 탭을 활용해 결과 깔끔하게 나누기
                tab1, tab2 = st.tabs(["📊 AI 용해도 예측", "🔬 물리화학적 상세 분석"])
                
                with tab1:
                    # 예측을 위한 데이터 변환 및 예측
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    features = np.array(fp).reshape(1, -1)
                    prediction = model.predict(features)
                    
                    st.subheader("용해도(LogS) 예측 결과")
                    st.success(f"이 분자의 예측 용해도 값(LogS)은 약 **{prediction[0]:.2f}**입니다.")
                    st.caption("※ LogS 값이 클수록(0에 가까울수록) 물에 잘 녹는 경향이 있음을 의미합니다.")
                
                with tab2:
                    st.subheader("분자 구조 특징 분석")
                    
                    # 분자량 계산
                    mw = Descriptors.MolWt(mol)
                    # 극성 표면적(TPSA) 계산
                    tpsa = Descriptors.TPSA(mol)
                    # 옥탄올-물 분배계수 (LogP)
                    logp = Descriptors.MolLogP(mol)
                    
                    st.metric(label="분자량 (Molecular Weight)", value=f"{mw:.2f} g/mol")
                    st.metric(label="극성 표면적 (TPSA)", value=f"{tpsa:.2f} Å²")
                    st.metric(label="분배계수 (LogP)", value=f"{logp:.2f}")
                    
                    # 간단한 극성 추정 가이드
                    if tpsa > 60 or logp < 0:
                        st.info("💡 **분자 성향 분석**: 이 분자는 극성 작용기를 포함하고 있어 상대적으로 **극성** 성향을 띨 가능성이 높습니다.")
                    else:
                        st.info("💡 **분자 성향 분석**: 이 분자는 탄화수소 사슬 등이 우세하여 상대적으로 **비극성** 성향을 띨 가능성이 높습니다.")
                    
                    # 교과 연계 팁
                    with st.expander("교과 연계 설명 보기"):
                        st.write("""
                        * **LogP**: 양수이면 소수성(비극성), 음수이면 친수성(극성) 경향을 보입니다.
                        * **TPSA**: 값이 클수록 분자 내에 극성 부분이 많음을 의미합니다. 물 분자와 수소 결합을 잘 형성하는지 예측할 수 있습니다.
                        """)
                        
        else:
            st.error("올바른 SMILES 형식이 아닙니다. 올바른 구조식을 입력해주세요.")

st.markdown("---")
st.info("💡 **팁**: 아스피린(CC(=O)Oc1ccccc1C(=O)O) 같은 복잡한 분자도 입력해 보세요!")
