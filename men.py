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
            model_name = "llama-3.3-70b-versatile"

            with st.status("Free multi-agent crew is working...", expanded=True) as status:
                
                # Step 1: Orchestration & Planning Agent
                st.write("Agent 1 (Orchestrator) is planning the research...")
                plan_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert research coordinator skilled at breaking down complex topics into clear, structured research plans."},
                        {"role": "user", "content": f"Create a comprehensive research plan outline for the topic: {topic}"}
                    ],
                    temperature=0.5
                )
                research_plan = plan_response.choices[0].message.content

                # Step 2: Literature Review / Fact-Finding Agent
                st.write("Agent 2 (Literature Reviewer) is gathering data and facts...")
                review_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a meticulous literature reviewer who extracts deep insights, factual data, and reliable information based on a research plan."},
                        {"role": "user", "content": f"Based on this plan, gather factual information and data points for '{topic}':\n\n{research_plan}"}
                    ],
                    temperature=0.5
                )
                gathered_data = review_response.choices[0].message.content

                # Step 3: Synthesis & Drafting Agent (Writer)
                st.write("Agent 3 (Writer) is synthesizing the final report...")
                writer_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert technical writer who turns research outlines and data points into polished, comprehensive markdown reports."},
                        {"role": "user", "content": f"Write a final, polished Markdown report on '{topic}' using the plan and gathered facts below:\n\nPlan:\n{research_plan}\n\nFacts:\n{gathered_data}"}
                    ],
                    temperature=0.5
                )
                final_report = writer_response.choices[0].message.content

                status.update(label="Research Completed successfully!", state="complete")

            st.subheader("Final Generated Report")
            st.markdown(final_report)

        except Exception as e:
            st.error(f"An error occurred during execution: {str(e)}")
