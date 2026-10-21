import os
import streamlit as st
from groq import Groq
import google.generativeai as genai
from openai import OpenAI

# ==========================================
# STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Multi-Provider Autonomous Researcher", layout="wide")
st.title("Multi-AI Autonomous Research Assistant")
st.write("Powered by Groq, Gemini, and OpenRouter")

# Sidebar for Multi-API Key Setup
with st.sidebar:
    st.header("API Key Configuration")
    groq_key = st.text_input("Groq API Key (Orchestration)", type="password")
    gemini_key = st.text_input("Gemini API Key (Analysis)", type="password")
    openrouter_key = st.text_input("OpenRouter API Key (Writing)", type="password")
    st.markdown("---")
    st.markdown("Multi-Model Pipeline Enabled")

topic = st.text_input("Enter your research topic:", value="")

if st.button("Start Multi-Provider Research"):
    if not (groq_key and gemini_key and openrouter_key):
        st.error("Please provide all three API keys in the sidebar to run the multi-provider pipeline.")
    elif not topic.strip():
        st.warning("Please enter a valid research topic.")
    else:
        try:
            with st.status("Multi-AI pipeline executing deep research...", expanded=True) as status:
                
                # ==========================================
                # AGENT 1: ORCHESTRATION (Groq - Unchanged)
                # ==========================================
                st.write("Agent 1 (Orchestrator) running on Groq...")
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

                # ==========================================
                # AGENT 2: LITERATURE REVIEW (Google Gemini - Unchanged)
                # ==========================================
                st.write("Agent 2 (Literature Reviewer) running on Google Gemini...")
                genai.configure(api_key=gemini_key)
                gemini_model = genai.GenerativeModel("gemini-3.8-flash")
                
                review_prompt = f"""
                You are a rigorous academic reviewer. Based on the outline below, generate deep analytical insights, data perspectives, and theoretical frameworks for '{topic}':
                
                Outline:
                {research_plan}
                """
                review_response = gemini_model.generate_content(review_prompt)
                gathered_data = review_response.text

                # ==========================================
                # AGENT 3: SYNTHESIS & DRAFTING (OpenRouter)
                # ==========================================
                st.write("Agent 3 (Writer) running via OpenRouter...")
                openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key
                )
                
                # Using an active, high-availability model ID route via OpenRouter
                writer_response = openrouter_client.chat.completions.create(
                    model="anthropic/claude-3.5-sonnet",
                    messages=[
                        {"role": "system", "content": "You are a professional academic author. Write a long, highly detailed, publication-grade research paper in Markdown format with full paragraphs, deep context, and exhaustive explanations—avoid short summaries."},
                        {"role": "user", "content": f"Write an extensive, fully fleshed-out research paper on '{topic}'. Use professional academic formatting, deep elaboration, detailed subsections, and synthesize the outline and granular points below:\n\nOutline:\n{research_plan}\n\nAnalytical Points:\n{gathered_data}"}
                    ],
                    temperature=0.4,
                    max_tokens=4096
                )
                final_report = writer_response.choices[0].message.content

                status.update(label="Comprehensive Research Completed Successfully!", state="complete")

            st.subheader("Final Publication-Grade Research Report")
            st.markdown(final_report)

        except Exception as e:
            st.error(f"An error occurred during multi-provider execution: {str(e)}")
