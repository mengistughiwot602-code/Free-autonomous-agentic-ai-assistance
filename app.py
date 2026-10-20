import os
import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM

# ==========================================
# STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Free Multi-AI Autonomous Researcher", layout="wide")
st.title("Free Multi-AI Autonomous Research Assistant")
st.write("Powered by GroqCloud, Google Gemini, and OpenRouter with zero search keys required!")

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
            st.write("Initializing LLM instances...")
            
            # Initialize CrewAI LLM objects
            groq_llm = LLM(model="groq/llama-3.3-70b-versatile")
            gemini_llm = LLM(model="gemini/gemini-1.5-flash")
            openrouter_llm = LLM(model="openrouter/meta-llama/llama-3-70b-instruct")

            st.write("Configuring specialized free model agents...")
            
            # Agent 1: Orchestration & Planning (Backstory included)
            orchestrator = Agent(
                role="Orchestrator",
                goal=f"Plan research on {topic}",
                backstory="You excel at task planning, logical flow organization, and structuring research steps efficiently.",
                llm=groq_llm,
                verbose=True
            )
            
            # Agent 2: Document Ingestion & Literature Review (Backstory included)
            reviewer = Agent(
                role="Literature Reviewer",
                goal=f"Gather data on {topic}",
                backstory="You leverage massive context windows to ingest technical information, extracting precise facts and data.",
                tools=[],
                llm=gemini_llm,
                verbose=True
            )
            
            # Agent 3: Synthesis, Critique & Drafting (Backstory included)
            writer = Agent(
                role="Writer",
                goal=f"Synthesize report on {topic}",
                backstory="You take analytical findings and craft polished, publication-grade technical Markdown reports.",
                llm=openrouter_llm,
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
