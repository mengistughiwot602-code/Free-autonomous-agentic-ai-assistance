import os
import streamlit as st
from groq import Groq
import google.generativeai as genai
from openai import OpenAI

# ==========================================
# STREAMLIT UI CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Multi-AI Autonomous Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dashboard Aesthetics
st.markdown("""
    <style>
        /* Main background and font styling */
        .main {
            background-color: #f8f9fa;
        }
        /* Header styling */
        h1 {
            color: #1e293b;
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        /* Subtext / Description */
        p {
            color: #475569;
            font-size: 1.05rem;
        }
        /* Sidebar container tweaks */
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            color: #f8fafc;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: #f8fafc !important;
        }
        /* Custom Button Styling */
        .stButton>button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3);
            transform: translateY(-1px);
        }
        /* Footer text styling */
        .footer-text {
            font-size: 0.85rem;
            color: #94a3b8;
            text-align: center;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/ffffff/artificial-intelligence.png", width=60)
    st.header("🔑 API Configuration")
    st.markdown("Fill in your required provider credentials below:")
    
    groq_key = st.text_input("Groq API Key (Orchestration)", type="password", help="Used for rapid planning & structuring")
    gemini_key = st.text_input("Gemini API Key (Analysis)", type="password", help="Used for deep literature review")
    openrouter_key = st.text_input("OpenRouter API Key (Writing)", type="password", help="Used for full academic report generation")
    
    st.markdown("---")
    st.markdown("<p class='footer-text'>Developed by DSstudents@bdu</p>", unsafe_allow_html=True)

# ==========================================
# MAIN PAGE CONTENT
# ==========================================
st.title("⚡ Multi-AI Autonomous Research Assistant")
st.markdown("This advanced multi-agent framework orchestrates Groq, Gemini, and OpenRouter to generate publication-grade long-form academic research papers.")

st.markdown("### 📝 Research Configuration")
topic = st.text_input("Enter your research topic:", placeholder="e.g., Impact of Climate Change on Sub-Saharan Agriculture")

st.markdown("")

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if st.button("🚀 Start Autonomous Research"):
    if not (groq_key and gemini_key and openrouter_key):
        st.error("⚠️ Please provide all three API keys in the sidebar before starting the research pipeline.")
    elif not topic.strip():
        st.warning("⚠️ Please enter a valid research topic.")
    else:
        try:
            with st.status("🔄 Multi-AI pipeline executing deep research...", expanded=True) as status:
                
                # Step 1: Groq Orchestration
                st.write("🤖 Agent 1 (Orchestrator) running on Groq (`openai/gpt-oss-20b`)...")
                groq_client = Groq(api_key=groq_key)
                
                plan_response = groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": "You are an elite academic research director. Create an exhaustive, multi-part chapter outline for a comprehensive research paper."},
                        {"role": "user", "content": f"Create a detailed 5-part structural outline for an extensive research paper on: {topic}"}
                    ],
                    temperature=0.4,
                    max_tokens=1500
                )
                research_plan = plan_response.choices[0].message.content

                # Step 2: Gemini Analysis
                st.write("🧠 Agent 2 (Literature Reviewer) running on Google Gemini (`gemini-3.8-flash`)...")
                genai.configure(api_key=gemini_key)
                gemini_model = genai.GenerativeModel("gemini-3.8-flash")
                
                review_prompt = f"""
                You are a rigorous academic reviewer. Based on the outline below, generate deep analytical insights, data perspectives, and theoretical frameworks for '{topic}':
                
                Outline:
                {research_plan}
                """
                review_response = gemini_model.generate_content(review_prompt)
                gathered_data = review_response.text

                # Step 3: OpenRouter Writing
                st.write("✍️ Agent 3 (Writer) running via OpenRouter (`openai/gpt-4o-mini`)...")
                openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key
                )
                
                writer_response = openrouter_client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional academic author. Write a long, highly detailed, publication-grade research paper in Markdown format with full paragraphs, deep context, and exhaustive explanations—avoid short summaries."},
                        {"role": "user", "content": f"Write an extensive, fully fleshed-out research paper on '{topic}'. Use professional academic formatting, deep elaboration, detailed subsections, and synthesize the outline and granular points below:\n\nOutline:\n{research_plan}\n\nAnalytical Points:\n{gathered_data}"}
                    ],
                    temperature=0.4,
                    max_tokens=4096
                )
                final_report = writer_response.choices[0].message.content

                status.update(label="✨ Comprehensive Research Completed Successfully!", state="complete")

            st.markdown("---")
            st.header("📄 Final Publication-Grade Research Report")
            st.markdown(final_report)

        except Exception as e:
            st.error(f"❌ An error occurred during multi-provider execution: {str(e)}")
