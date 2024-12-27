from langchain_openai import ChatOpenAI
from collections import defaultdict
from typing import Dict, Tuple
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

class InterviewMemory:
    def __init__(self):
        self.field_memory = defaultdict(list)
        self.current_responses = defaultdict(str)
    
    def add_response(self, field: str, response: str):
        self.field_memory[field].append(response)
        # Combine all responses for this field
        self.current_responses[field] = " ".join(self.field_memory[field])
    
    def get_field_history(self, field: str) -> str:
        return self.current_responses[field]

# Detailed form structure with satisfaction tracking
interview_form = {
    "name": {"value": "", "satisfaction": 0},
    "current_role": {"value": "", "satisfaction": 0},
    "years_of_experience": {"value": "", "satisfaction": 0},
    "technical_skills": {"value": [], "satisfaction": 0},
    "project_experience": {"value": "", "satisfaction": 0},
    "motivation": {"value": "", "satisfaction": 0},
    "preferred_work_environment": {"value": "", "satisfaction": 0},
}

# Comprehensive field requirements and expected information
FIELD_REQUIREMENTS = {
    "name": {
        "description": "Full name of the candidate",
        "expected": "First and last name",
        "follow_up_questions": [
            "Could you spell your full name for me?",
            "Do you go by any other names professionally?"
        ]
    },
    "current_role": {
        "description": "Current job position and main responsibilities",
        "expected": "Job title, company, key responsibilities, team size, main achievements",
        "follow_up_questions": [
            "What are your main responsibilities in this role?",
            "How large is the team you work with?",
            "What have been your key achievements in this position?"
        ]
    },
    "years_of_experience": {
        "description": "Total relevant work experience",
        "expected": "Years of total experience, years in current field, career progression",
        "follow_up_questions": [
            "How long have you been working in this field specifically?",
            "Could you briefly outline your career progression?",
            "What different roles have you held during your career?"
        ]
    },
    "technical_skills": {
        "description": "Technical abilities and proficiency levels",
        "expected": "List of skills with proficiency levels (beginner/intermediate/expert), recent usage",
        "follow_up_questions": [
            "Could you rate your proficiency in each skill mentioned?",
            "How recently have you used these skills?",
            "What projects have you completed using these skills?"
        ]
    },
    "project_experience": {
        "description": "Significant projects and achievements",
        "expected": "Project descriptions, role, technologies used, outcomes, challenges overcome",
        "follow_up_questions": [
            "What was your specific role in these projects?",
            "What challenges did you face and how did you overcome them?",
            "What were the measurable outcomes of these projects?"
        ]
    },
    "motivation": {
        "description": "Career goals and motivation for the position",
        "expected": "Short-term and long-term goals, interest in the position, alignment with career path",
        "follow_up_questions": [
            "What interests you most about this position?",
            "Where do you see yourself in 5 years?",
            "How does this role align with your career goals?"
        ]
    },
    "preferred_work_environment": {
        "description": "Work style and preferred environment",
        "expected": "Preferred work style, team dynamics, company culture, work-life balance",
        "follow_up_questions": [
            "What type of company culture do you thrive in?",
            "How do you prefer to collaborate with team members?",
            "What management style works best for you?"
        ]
    }
}

def create_evaluation_prompt(response: str, field: str) -> str:
    """
    Create a detailed prompt for response evaluation
    """
    return f"""
    You are an expert interviewer evaluating candidate responses.
    
    Field: {field}
    Expected information: {FIELD_REQUIREMENTS[field]['expected']}
    Candidate's response: "{response}"
    
    Please evaluate the response and provide:
    1. Satisfaction score (1-10) based on completeness and relevance
    2. Analysis of missing or unclear information
    3. Specific follow-up question if needed
    
    Return your evaluation in this exact JSON format:
    {{
        "satisfaction_score": <int>,
        "missing_info": "<string>",
        "follow_up_question": "<string>",
        "analysis": "<string>"
    }}
    """

def is_negative_response(response: str) -> bool:
    """
    Check if the response is a clear negative
    """
    negative_indicators = [
        "no", "none", "nothing", "don't have", "do not have",
        "nothing comes to mind", "haven't done any"
    ]
    response_lower = response.lower().strip()
    return any(indicator in response_lower for indicator in negative_indicators)

def evaluate_response(client, response: str, field: str, memory: InterviewMemory) -> Dict:
    try:
        # Get complete history for this field
        complete_response = memory.get_field_history(field)
        
        # Create the evaluation prompt
        evaluation_prompt = f"""
        You are evaluating a response for the field: {field}
        Expected information: {FIELD_REQUIREMENTS[field]['expected']}
        Complete response history: {complete_response}
        Latest response: {response}

        Please evaluate the complete response history and provide:
        1. A satisfaction score (1-10)
        2. A brief analysis
        3. Any missing information
        4. A follow-up question if needed

        Format your response as JSON:
        {{
            "satisfaction_score": <score>,
            "analysis": "<brief analysis>",
            "missing_info": "<list missing info or 'none'>",
            "follow_up_question": "<question to get missing info>"
        }}
        """

        # Get completion from OpenAI
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert interviewer evaluating responses."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.7
        )

        # Extract the response
        evaluation = json.loads(completion.choices[0].message.content)
        
        # Add default values if any key is missing
        evaluation.setdefault("satisfaction_score", 5)
        evaluation.setdefault("analysis", "Analysis not provided")
        evaluation.setdefault("missing_info", "None")
        evaluation.setdefault("follow_up_question", FIELD_REQUIREMENTS[field]['follow_up_questions'][0])
        
        return evaluation

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return {
            "satisfaction_score": 5,
            "analysis": "Error occurred during analysis",
            "missing_info": "Error in evaluation",
            "follow_up_question": FIELD_REQUIREMENTS[field]['follow_up_questions'][0] if field in FIELD_REQUIREMENTS else "Could you please provide more details?",
            "skip_topic": False
        }

def determine_next_question(interview_form: Dict, current_field: str) -> Tuple[str, str]:
    """
    Determine the next question based on current form state
    """
    # Get list of all fields
    fields = list(interview_form.keys())
    
    # If we have a current field and it's complete, move to next field
    if current_field:
        try:
            current_index = fields.index(current_field)
            next_field = fields[current_index + 1]  # Get next field
            return (next_field, f"Let's talk about your {next_field.replace('_', ' ')}. {FIELD_REQUIREMENTS[next_field]['follow_up_questions'][0]}")
        except IndexError:
            # Only return None if we've really reached the end
            return (None, "We've covered everything I needed to know. Is there anything else you'd like to add?")
    
    # If no current field, find the first incomplete field
    for field in fields:
        if interview_form[field]["satisfaction"] < 7:
            return (field, FIELD_REQUIREMENTS[field]['follow_up_questions'][0])
    
    # If all fields are complete
    return (None, "We've covered everything I needed to know. Is there anything else you'd like to add?")

def save_interview_state(interview_data: Dict, filename: str = None, current_field: str = None):
    """
    Save the current interview state to a JSON file
    """
    # Create a structured output
    output = {
        "interview_timestamp": datetime.now().isoformat(),
        "interview_data": {}
    }
    
    # Process each field
    for field, data in interview_data.items():
        output["interview_data"][field] = {
            "value": data["value"],
            "satisfaction_score": data["satisfaction"],
            "status": "skipped" if data["satisfaction"] == 0 else 
                     "complete" if data["satisfaction"] >= 7 else 
                     "incomplete"
        }
    
    # Generate filename if not provided
    if filename is None:
        filename = f"interview_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Only print debug info if we have a current field
    if current_field:
        print(f"\n[Debug] JSON updated - Latest state for {current_field}:")
        print(json.dumps(output["interview_data"][current_field], indent=2))
    
    return filename

def conduct_interview(llm: ChatOpenAI):
    memory = InterviewMemory()
    current_field = "name"
    current_question = "Could you please tell me your full name?"
    
    # Create initial JSON file
    json_filename = save_interview_state(interview_form)
    print(f"\nInterview state file created: {json_filename}")
    
    while True:
        print("\n" + "="*50)
        print(f"Topic: {current_field.replace('_', ' ').title()}")
        print("="*50)
        print("\nInterviewer:", current_question)
        
        response = input("\nInterviewee: ")
        if response.lower() == 'exit':
            break
        
        # Evaluate response with memory
        evaluation = evaluate_response(llm, response, current_field, memory)
        
        # Handle negative responses
        if evaluation.get("skip_topic", False):
            print("\n[Interviewer Notes]")
            print(f"Topic skipped: {evaluation['analysis']}")
            # Update form to mark this field as not applicable
            interview_form[current_field]["value"] = "No experience reported"
            interview_form[current_field]["satisfaction"] = 0
            
            # Save updated state and print debug info
            print(f"\n[Debug] Updating JSON after skipping {current_field}")
            save_interview_state(interview_form, json_filename, current_field)
            
            # Find next field in sequence
            fields = list(interview_form.keys())
            try:
                current_index = fields.index(current_field)
                next_field = fields[current_index + 1]
                current_field = next_field
                current_question = f"Let's talk about your {next_field.replace('_', ' ')}. {FIELD_REQUIREMENTS[next_field]['follow_up_questions'][0]}"
                continue
            except (ValueError, IndexError):
                break
        
        # Update form with combined information
        interview_form[current_field]["value"] = memory.get_field_history(current_field)
        interview_form[current_field]["satisfaction"] = evaluation["satisfaction_score"]
        
        # Save updated state and print debug info
        print(f"\n[Debug] Updating JSON after response to {current_field}")
        save_interview_state(interview_form, json_filename)
        
        # Save updated state to file
        save_interview_state(interview_form, json_filename, current_field)
        # Show interviewer notes
        print("\n[Interviewer Notes]")
        print(f"Satisfaction Score: {evaluation['satisfaction_score']}/10")
        print(f"Summary: {evaluation['summary']}")
        print(f"Analysis: {evaluation['analysis']}")
        
        if evaluation["satisfaction_score"] < 7:
            print(f"Missing Information: {evaluation['missing_info']}")
            current_question = evaluation["follow_up_question"]
        else:
            current_field, current_question = determine_next_question(interview_form, current_field)
            if current_field is None:
                break
    
    print("\nFinal interview state saved to:", json_filename)
    # Print final state for verification
    with open(json_filename, 'r', encoding='utf-8') as f:
        final_state = json.load(f)
        print("\nFinal JSON content:")
        print(json.dumps(final_state, indent=2))
    
    return json_filename

def main():
    llm = ChatOpenAI(temperature=0.7, model="gpt-4")
    conduct_interview(llm)

if __name__ == "__main__":
    main()