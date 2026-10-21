import os
import streamlit as st
from crewai import Agent, Crew, Process, Task

# ==========================================
# STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Free Multi-AI Autonomous Researcher", layout="wide")
st.title("Free Multi-AI Autonomous Research Assistant")
st.write("Powered by GroqCloud with zero search keys required!")

# Sidebar for API Keys Setup
with st.sidebar:
    st.header("Free API Keys Setup")
    groq_key = st.text_input("Groq API Key", type="password")
    gemini_key = st.text_input("Google Gemini API Key", type="password")
    openrouter_key = st.text_input("OpenRouter API Key", type="password")
    st.markdown("---")
    st.markdown("Built with [CrewAI](https://crewai.com)")

topic = st.text_input("Enter your research topic:", value="")

if st.button("Start Free Autonomous Research"):
    if not all([groq_key, gemini_key, openrouter_key]):
        st.error("Please provide all required free API keys in the sidebar.")
    elif not topic.strip():
        st.warning("Please enter a valid research topic.")
    else:
        # Set environment variables securely from user inputs
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["OPENROUTER_API_KEY"] = openrouter_key

        with st.status("Free multi-agent crew is working...", expanded=True) as status:
            st.write("Configuring specialized free model agents...")
            
            # Agent 1: Orchestration & Planning
            orchestrator = Agent(
                role="Orchestrator",
                goal=f"Plan research on {topic}",
                backstory="You are an expert research coordinator skilled at breaking down complex topics into clear, structured research plans.",
                llm="groq/llama-3.3-70b-versatile",
                verbose=True
            )
            
            # Agent 2: Document Ingestion & Literature Review
            reviewer = Agent(
                role="Literature Reviewer",
                goal=f"Gather data on {topic}",
                backstory="You are a meticulous literature reviewer who extracts deep insights, factual data, and reliable information.",
                tools=[],
                llm="groq/llama-3.3-70b-versatile",
                verbose=True
            )
            
            # Agent 3: Synthesis, Critique & Drafting
            writer = Agent(
                role="Writer",
                goal=f"Synthesize report on {topic}",
                backstory="You are an expert technical writer who turns rough outlines and data points into polished, comprehensive markdown reports.",
                llm="groq/llama-3.3-70b-versatile",
                verbose=True
            )

            st.write("Assembling tasks and executing workflow...")
            t1 = Task(
                description=f"Plan research for {topic}",
                expected_output="Plan outline",
                agent=orchestrator
            )
            t2 = Task(
                description=f"Search facts for {topic}",
                expected_output="Fact list",
                agent=reviewer
            )
            t3 = Task(
                description=f"Draft report on {topic}",
                expected_output="Markdown report",
                agent=writer
            )

            crew = Crew(
                agents=[orchestrator, reviewer, writer],
                tasks=[t1, t2, t3],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff(inputs={"topic": topic})
            status.update(label="Research Completed successfully!", state="complete")

        st.subheader("Final Generated Report")
        st.markdown(result)
