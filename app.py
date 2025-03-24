from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY=st.secrets["DEEPSEEK_API_KEY"]

# System prompts
INTERVIEW_PROMPT = (
    "# Role: Intellegent Adaptive Meeting Assistant\n"
    "Goal: Gather structured information for meeting minutes while avoiding redundancy.\n\n"
    "## Workflow\n"
    "1. Pre-Process Notes:\n"
    "   - Extract answers from user-provided notes upfront (e.g., \"Company: XYZ Corp, Challenges: Scaling\" → auto-populate fields).\n"
    "   - Skip questions already answered.\n\n"
    "2. Core Questions (Ask Only If Missing):\n"
    "   - \"What is the company name?\"\n"
    "   - \"Who attended the meeting?\"\n"
    "   - \"Where/long did it take place?\"\n"
    "   - \"Employees/management levels?\"\n\n"
    "3. Dynamic Exploration:\n"
    "   - Ask ONE question at a time from these topics only if unanswered:\n"
    "     - Strategic goals → \"What's the #1 priority for Q3?\"\n"
    "     - Development focus → \"Which initiatives need acceleration?\"\n"
    "     - Challenges → \"What's the biggest roadblock?\"\n"
    "     - Action items → \"Who owns [task] and by when?\"\n"
    "     - Follow-up timing → \"When should we review progress?\"\n\n"
    "4. Adaptive Rules:\n"
    "   - Before asking ANY question:\n"
    "     - Check conversation history for answers.\n"
    "     - If answer exists:\n"
    "       - ✔️ Confirm: \"You mentioned [X]. Is this correct?\"\n"
    "       - ➡️ Clarify if ambiguous: \"For [X], did you mean [interpretation]?\"\n"
    "     - If incomplete → ask follow-ups: \"Can you elaborate on [specific detail]?\"\n\n"
    "5. Format:\n"
    "   - Conversational but professional.\n"
    "   - Always summarize key points before moving to next topic.\n"
    "   - Example flow:\n"
    "     > User: \"Attendees: John (CEO), Sarah (CTO)\"\n"
    "     > Assistant: \"Got it. Next: Could you share the top strategic goal discussed?\"\n\n"
    "Note: Always cross-check that all needed questions are answered. If all questions have been answered, inform the user that all questions are done and prompt them to click on the 'Generate MOM Generation' button to get their MOM.\n"
)

MOM_PROMPT = (
    "You are a professional meeting assistant tasked with generating comprehensive Meeting Minutes (MoM) that a consultant can immediately use for follow-ups. Based on the given interview data provided, generate a final MoM document with the following sections and in a clear, business-friendly format:\n\n"
    "---\n\n"
    "## Meeting Minutes (MoM)\n\n"
    "### 1. Meeting Overview\n"
    "- *Company Name:* [Extract from data]\n"
    "- *Meeting Date & Time:* [If available]\n"
    "- *Location:* [Extract from data]\n"
    "- *Duration:* [Extract from data]\n"
    "- *Participants:* [List all names and roles]\n\n"
    "### 2. Meeting Objective\n"
    "- Provide a concise summary of the meeting's purpose (e.g., discussing training needs, leadership development, or strategic planning).\n\n"
    "### 3. Discussion Summary\n"
    "- *Key Topics:*  \n"
    "  Summarize the main discussion points. Include any specific areas such as:\n"
    "  - Strategic goals and development focus\n"
    "  - Target groups for development and current challenges\n"
    "  - Existing training programs and preferred learning formats\n"
    "- *Additional Context:*  \n"
    "  Include any notable insights, pain points, or suggestions mentioned during the discussion.\n\n"
    "### 4. Action Items & Follow-Up\n"
    "- *Action Items:*  \n"
    "  List each agreed-upon action with a brief description.\n"
    "- *Responsibilities:*  \n"
    "  Specify who is responsible for each action.\n"
    "- *Follow-Up:*  \n"
    "  Note the agreed timeline or date for checking progress.\n\n"
    "### 5. Additional Notes\n"
    "- Add any extra information or clarifications provided that do not fit in the sections above.\n\n"
    "---\n\n"
    "Using the raw interview data below, generate the final Meeting Minutes (MoM) in the above format. You can also add something by yourself if its important for consultant. "
    "Ensure that the output is neatly formatted with clear headings and bullet points, includes only the necessary details as discussed, and omits any extraneous information.\n\n"
    "Generate the final Meeting Minutes (MoM) now."
)

def create_chat():
    # Get API key from environment variable
    # openai_api_key = os.getenv('DEEPSEEK_API_KEY')
    # if not openai_api_key:
    #     raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

    chat = ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-reasoner",
        temperature=1,
        openai_api_key=DEEPSEEK_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(INTERVIEW_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    class ChatHistory:
        def __init__(self):
            self.conversation = None
            self.message_history = None
            self._initialize()

        def _initialize(self):
            class CustomChatMessageHistory(BaseChatMessageHistory):
                def __init__(self):
                    self.messages = []

                def add_message(self, message):
                    self.messages.append(message)

                def clear(self):
                    self.messages = []

            message_history = CustomChatMessageHistory()
            chain = prompt | chat
            
            self.conversation = RunnableWithMessageHistory(
                chain,
                lambda session_id: message_history,
                input_messages_key="input",
                history_messages_key="history"
            )
            self.message_history = message_history

        def invoke(self, input_text=None, **kwargs):
            # Handle both direct input and keyword arguments
            input_content = input_text if input_text is not None else kwargs.get('input', '')
            
            response = self.conversation.invoke(
                {"input": input_content},
                config={"configurable": {"session_id": "default"}}
            )
            
            # Store messages in history
            self.message_history.add_message(HumanMessage(content=input_content))
            self.message_history.add_message(AIMessage(content=response.content))
            
            # Return just the content string
            return response.content

    return ChatHistory()

def create_mom_chain():
    # Get API key from environment variable
    # openai_api_key = os.getenv('OPENAI_API_KEY')
    # if not openai_api_key:
    #     raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Initialize ChatOpenAI with DeepSeek configuration
    chat = ChatOpenAI(
        base_url="https://api.deepseek.com/v1",  # Deepseek's API endpoint
        model="deepseek-reasoner",
        temperature=1,
        openai_api_key=DEEPSEEK_API_KEY         # Your Deepseek API key
    )

    # Create prompt template for MoM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MOM_PROMPT),
        HumanMessagePromptTemplate.from_template("Here is the interview transcript:\n\n{interview_history}")
    ])

    # Create MoM chain without memory since we just need to generate once
    chain = LLMChain(
        llm=chat,
        prompt=prompt,
        verbose=True
    )
    
    return chain

def generate_mom(message_history):
    # Format conversation history as a clear Q&A transcript
    interview_history = ""
    for i in range(0, len(message_history.messages), 2):
        if i + 1 < len(message_history.messages):
            question = message_history.messages[i].content
            answer = message_history.messages[i + 1].content
            interview_history += f"Q: {question}\nA: {answer}\n\n"
    
    # Create and run MoM chain with interview history
    mom_chain = create_mom_chain()
    mom = mom_chain.run(interview_history=interview_history)
    return mom

def main():
    chat_manager = create_chat()
    
    print("Welcome to the Meeting Analysis Chatbot!")
    print("Type 'hi' to start conversation or Type 'quit' to end the conversation\n")
    
    while True:
        user_input = input("\nYou: ").strip().lower()
        
        if user_input == 'quit':
            print("\nThank you for using the Meeting Analysis Chatbot!")
            break
        
        # Get response as simple string
        response = chat_manager.invoke(input_text=user_input)
        print(f"\nBot: {response}")
        
        if user_input == 'generate mom':
            print("\nGenerating Meeting Minutes...\n")
            mom = generate_mom(chat_manager.message_history)
            print("=== Meeting Minutes ===")
            print(mom)
            print("=====================")
            continue

if __name__ == "__main__":
    main()
