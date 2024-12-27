from openai import OpenAI
import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from utils import (
    InterviewMemory, 
    is_negative_response, 
    evaluate_response, 
    FIELD_REQUIREMENTS,
    load_chat_history,
    save_chat_history
)
load_dotenv()

def initialize_session_state():
    # Try to load existing chat history
    if "messages" not in st.session_state or "interview_form" not in st.session_state or "memory" not in st.session_state:
        messages, interview_form, memory = load_chat_history()
        
        if messages:  # If we found saved data
            st.session_state.messages = messages
            st.session_state.interview_form = interview_form
            st.session_state.memory = memory
        else:  # Initialize new session
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Could you please tell me your full name?"
            }]
            st.session_state.interview_form = {
                field: {"value": "", "responses": [], "satisfaction": 0}
                for field in FIELD_REQUIREMENTS.keys()
            }
            st.session_state.memory = InterviewMemory()

    if "current_field" not in st.session_state:
        st.session_state.current_field = "name"
    if "show_summary" not in st.session_state:
        st.session_state.show_summary = True

def display_interview_summary():
    with st.sidebar:
        st.markdown("## 📋 Interview Summary")
        
        # Overall progress
        total_fields = len(st.session_state.interview_form)
        completed_fields = sum(1 for field in st.session_state.interview_form.values() 
                             if field["satisfaction"] >= 7)
        st.progress(completed_fields / total_fields)
        st.markdown(f"**Progress:** {completed_fields}/{total_fields} topics completed")
        
        # Detailed summary for each field
        st.markdown("### Detailed Responses")
        for field, data in st.session_state.interview_form.items():
            satisfaction = data["satisfaction"]
            
            # Determine status emoji
            if satisfaction >= 7:
                status = "✅"
            elif satisfaction > 0:
                status = "⚠️"
            else:
                status = "❌"
            
            # Create expandable section for each field
            with st.expander(f"{status} {field.replace('_', ' ').title()} ({satisfaction}/10)"):
                if data["value"]:
                    st.markdown("**Responses:**")
                    if isinstance(data["responses"], list):
                        for i, response in enumerate(data["responses"], 1):
                            st.markdown(f"- {response}")
                    else:
                        st.markdown(data["value"])
                else:
                    st.markdown("*No response provided*")

def get_next_field(client, prompt):
    # check for negative response
    if is_negative_response(client, prompt):
        fields = list(st.session_state.interview_form.keys())
        current_index = fields.index(st.session_state.current_field)
        next_field = fields[current_index + 1] if current_index + 1 < len(fields) else None
        
        if next_field:
            st.session_state.current_field = next_field
            next_question = FIELD_REQUIREMENTS[next_field]["follow_up_questions"][0]
            response = f"I understand. Let's move on to your {next_field.replace('_', ' ')}. {next_question}"
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            response = "Thank you for your time. We've completed all topics!"
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        return next_field, response

    # evaluate response
    try: 
        evaluation = evaluate_response(
            client, 
            prompt, 
            st.session_state.current_field, 
            st.session_state.memory
        )

        # Update form with complete history
        st.session_state.interview_form[st.session_state.current_field].update({
            "value": st.session_state.memory.get_field_history(st.session_state.current_field),
            "responses": st.session_state.memory.get_all_responses(st.session_state.current_field),
            "satisfaction": evaluation["satisfaction_score"]
        })         

        if evaluation["satisfaction_score"] >= 7:
            fields = list(st.session_state.interview_form.keys())
            current_index = fields.index(st.session_state.current_field)
            next_field = fields[current_index + 1] if current_index + 1 < len(fields) else None
            
            if next_field:
                response = f"Great! Let's move on to your {next_field.replace('_', ' ')}. {FIELD_REQUIREMENTS[next_field]['follow_up_questions'][0]}"
            else:
                response = "Thank you for your time. We've completed all topics!"
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            return next_field, response
        
        else:
            response = evaluation["follow_up_question"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            return st.session_state.current_field, response
    
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        return None, str(e)

def main():
    initialize_session_state()

    # Show summary if enabled
    if st.session_state.show_summary:
        display_interview_summary()

    # Main chat area
    st.title("AI Interviewer")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your response"):
        # add user message to memory
        st.session_state.memory.add_response(st.session_state.current_field, prompt)

        # add user message to messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # process the response
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        next_field, response = get_next_field(client, prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)

        if next_field:
            st.session_state.current_field = next_field

        # Save chat history after each interaction
        save_chat_history(
            st.session_state.messages,
            st.session_state.interview_form,
            st.session_state.memory
        )

        # Force a rerun to update the display
        st.rerun()

if __name__ == "__main__":
    main()