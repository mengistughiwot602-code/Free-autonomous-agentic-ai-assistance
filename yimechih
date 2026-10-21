import os
import streamlit as st
from groq import Groq

# ==========================================
# STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Free Multi-AI Autonomous Researcher", layout="wide")
st.title("Free Multi-AI Autonomous Research Assistant")
st.write("Powered by GroqCloud with zero search keys required!")

# Sidebar for API Key Setup
with st.sidebar:
    st.header("Free API Key Setup")
    groq_key = st.text_input("Groq API Key", type="password")
    st.markdown("---")
    st.markdown("Built with Groq & Streamlit")

topic = st.text_input("Enter your research topic:", value="")

if st.button("Start Free Autonomous Research"):
    if not groq_key:
        st.error("Please provide your Groq API key in the sidebar.")
    elif not topic.strip():
        st.warning("Please enter a valid research topic.")
    else:
        try:
            # Initialize the official Groq client
            client = Groq(api_key=groq_key)
            model_name = "openai/gpt-oss-20b"

            with st.status("Deep research workflow in progress...", expanded=True) as status:
                
                # Step 1: Orchestration & Detailed Planning
                st.write("Agent 1 (Orchestrator) is designing a deep structural outline...")
                plan_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an elite academic research director. Create an exhaustive, multi-part chapter outline for a comprehensive research paper."},
                        {"role": "user", "content": f"Create a detailed 5-part structural outline for an extensive research paper on: {topic}"}
                    ],
                    temperature=0.4,
                    max_tokens=1500
                )
                research_plan = plan_response.choices[0].message.content

                # Step 2: In-Depth Literature Analysis & Fact-Finding
                st.write("Agent 2 (Literature Reviewer) is compiling deep insights and analytical arguments...")
                review_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a rigorous academic reviewer. Extract deep, granular technical insights, data perspectives, and theoretical frameworks for each section of the outline."},
                        {"role": "user", "content": f"Based on this outline, generate comprehensive analytical breakdowns, data points, and technical background for '{topic}':\n\n{research_plan}"}
                    ],
                    temperature=0.4,
                    max_tokens=2048
                )
                gathered_data = review_response.choices[0].message.content

                # Step 3: Comprehensive Synthesis & Full Report Generation
                st.write("Agent 3 (Writer) is compiling and drafting the full publication-grade report...")
                writer_response = client.chat.completions.create(
                    model=model_name,
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
            st.error(f"An error occurred during execution: {str(e)}")
