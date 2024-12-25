from openai import OpenAI
import streamlit as st
import os
from main import InterviewMemory, evaluate_response, determine_next_question, interview_form, FIELD_REQUIREMENTS
import json

def is_negative_response(response: str) -> bool:
    """
    Check if the response is a clear negative
    """
    negative_indicators = [
        "no", "none", "nothing", "don't have", "do not have",
        "nothing comes to mind", "haven't done any", "i don't",
        "no experience", "no projects"
    ]
    response_lower = response.lower().strip()
    return any(indicator in response_lower for indicator in negative_indicators)

def initialize_session_state():
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'memory' not in st.session_state:
        st.session_state.memory = InterviewMemory()
    if 'current_field' not in st.session_state:
        st.session_state.current_field = "name"
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Could you please tell me your full name?"
        })
    if 'interview_form' not in st.session_state:
        st.session_state.interview_form = interview_form.copy()
    if 'agent_processes' not in st.session_state:
        st.session_state.agent_processes = []

def process_response(client, prompt: str):
    # Check for negative response first
    if is_negative_response(prompt):
        st.session_state.agent_processes.append({
            "agent": "Documentation Agent",
            "action": "Analyzing response",
            "field": st.session_state.current_field,
            "status": "complete",
            "result": {
                "satisfaction": 0,
                "analysis": "Candidate has no experience in this area",
                "missing_info": "None - Topic skipped"
            }
        })
        
        # Update form to mark this field as not applicable
        st.session_state.interview_form[st.session_state.current_field]["value"] = "No experience reported"
        st.session_state.interview_form[st.session_state.current_field]["satisfaction"] = 0
        
        # Move to next field
        fields = list(st.session_state.interview_form.keys())
        current_index = fields.index(st.session_state.current_field)
        next_field = fields[current_index + 1] if current_index + 1 < len(fields) else None
        
        if next_field:
            st.session_state.current_field = next_field
            return f"I understand. Let's move on to your {next_field.replace('_', ' ')}. {FIELD_REQUIREMENTS[next_field]['follow_up_questions'][0]}"
        else:
            return "Thank you, we've completed all the topics!"

    # Add user message to memory
    st.session_state.memory.add_response(st.session_state.current_field, prompt)
    
    # Log processing status
    st.session_state.agent_processes.append({
        "agent": "Documentation Agent",
        "action": "Analyzing response",
        "field": st.session_state.current_field,
        "status": "processing"
    })
    
    try:
        # Evaluate response
        evaluation = evaluate_response(
            client, 
            prompt, 
            st.session_state.current_field, 
            st.session_state.memory
        )
        
        # Update agent status
        st.session_state.agent_processes[-1].update({
            "status": "complete",
            "result": {
                "satisfaction": evaluation["satisfaction_score"],
                "analysis": evaluation["analysis"],
                "missing_info": evaluation.get("missing_info", "None")
            }
        })
        
        # Update form with combined information
        st.session_state.interview_form[st.session_state.current_field]["value"] = st.session_state.memory.get_field_history(st.session_state.current_field)
        st.session_state.interview_form[st.session_state.current_field]["satisfaction"] = evaluation["satisfaction_score"]
        
        # Handle response based on evaluation
        if evaluation["satisfaction_score"] >= 7:
            fields = list(st.session_state.interview_form.keys())
            current_index = fields.index(st.session_state.current_field)
            next_field = fields[current_index + 1] if current_index + 1 < len(fields) else None
            
            if next_field:
                st.session_state.current_field = next_field
                return f"Great! Let's move on to your {next_field.replace('_', ' ')}. {FIELD_REQUIREMENTS[next_field]['follow_up_questions'][0]}"
            else:
                return "Thank you, we've completed all the topics!"
        else:
            return evaluation["follow_up_question"]
        
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        return FIELD_REQUIREMENTS[st.session_state.current_field]['follow_up_questions'][0]

def main():
    st.set_page_config(layout="wide")
    initialize_session_state()
    
    # Sidebar configuration
    st.sidebar.title("💡 Interview Progress")
    
    # Current Topic Section in sidebar
    if st.session_state.current_field:
        st.sidebar.markdown("### 📍 Current Topic")
        st.sidebar.info(st.session_state.current_field.replace('_', ' ').title())
    
    # Progress Tracker in sidebar
    st.sidebar.markdown("### 📊 Progress Tracker")
    for field, data in st.session_state.interview_form.items():
        if data["satisfaction"] == 0:
            status = "⏭️"  # Skipped
        elif data["satisfaction"] >= 7:
            status = "✅"  # Complete
        else:
            status = "⏳"  # In progress
        st.sidebar.write(f"{status} {field.replace('_', ' ').title()}")
    
    # Agent Activities Section in sidebar
    st.sidebar.markdown("### 🔄 Recent Activities")
    for process in st.session_state.agent_processes[-3:]:  # Show only last 3 activities
        with st.sidebar.expander(f"🔹 {process['action']}", expanded=False):
            if process['status'] == 'processing':
                st.info(f"Status: {process['status']}")
            else:
                st.success(f"Status: {process['status']}")
            if process['status'] == 'complete' and 'result' in process:
                for key, value in process['result'].items():
                    st.write(f"**{key.title()}:** {value}")
    
    # Main chat area
    st.title("🤖 AI Interview Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your response here..."):
        # Add user message to display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("Please set your OpenAI API key")
            return
        
        client = OpenAI(api_key=api_key)
        
        # Process response and get next question
        next_message = process_response(client, prompt)
        
        # Display assistant response
        st.session_state.messages.append({"role": "assistant", "content": next_message})
        with st.chat_message("assistant"):
            st.markdown(next_message)

if __name__ == "__main__":
    main()