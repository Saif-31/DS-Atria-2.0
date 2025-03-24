import streamlit as st
import os
from app import create_chat, generate_mom


# Set Deepseek API key from Streamlit secrets
os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]

# Add custom CSS for button color
st.markdown("""
    <style>
        .stButton > button {
            background-color: #00008B;
            color: white;
        }
        .stButton > button:hover {
            background-color: #0000CD;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Check for API key
if not os.environ.get("DEEPSEEK_API_KEY"):
    st.error("DeepSeek API key is not set!")
    st.stop()

# Sidebar Guide
with st.sidebar:
    # Generate MoM button is always available
    generate_mom_clicked = st.button("Generate MoM", use_container_width=True, key="sidebar_generate")
    
    st.markdown("---")
    st.header("How to Use")
    st.markdown("""
    1. **Start the Conversation**
        * Type 'hi' to begin the interview
        * The bot will ask you essential questions about the meeting
    
    2. **During the Interview**
        * Answer each question clearly and concisely
        * Provide all relevant details requested
        * The bot will ask follow-up questions as needed
    
    3. **Completing the Interview**
        * The bot will confirm when all necessary information is gathered
        * You can add any additional information if needed
    
    4. **Generating Minutes**
        * Once the interview is complete, click 'Generate Minutes'
        * The bot will create a formatted Meeting Minutes document
    
    5. **End Session**
        * Type 'quit' to end the conversation
    """)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = create_chat()
    st.session_state.messages = []
    # Remove interview_completed flag since we're letting user control the flow
    welcome_msg = "Welcome to the Meeting Analysis Chatbot! Complete your interview and click on 'Generate MoM' button on your top left corner when ready."
    st.session_state.messages.extend([
        {"role": "assistant", "content": welcome_msg}
    ])

# Title
st.title("Meeting Minutes Bot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Generate MoM button click
if generate_mom_clicked:
    if len(st.session_state.messages) > 1:  # Check if there's any conversation
        with st.spinner("Generating Meeting Minutes..."):
            mom = generate_mom(st.session_state.conversation.message_history)
            # Display MoM in chat
            with st.chat_message("assistant"):
                st.markdown("### Generated Meeting Minutes")
                st.markdown(mom)
            # Add MoM to message history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"### Generated Meeting Minutes\n\n{mom}"
            })
            # Force update
            st.rerun()
    else:
        with st.chat_message("assistant"):
            st.warning("Please complete your interview first before generating minutes.")

# Chat input with unique key
if prompt := st.chat_input("Type your message here...", key="chat_input"):
    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response from first agent only
    with st.chat_message("assistant"):
        response = st.session_state.conversation.invoke(input=prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Force a rerun to update the chat display
    st.rerun()
