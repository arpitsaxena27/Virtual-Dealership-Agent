import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Virtual Dealership Agent", page_icon="🚗")
st.title("🚗 Virtual Dealership Assistant")
st.markdown("Ask me anything about the vehicle based on the official manual!")

# 2. Initialize Chat History
# Streamlit reruns the script on every interaction, so we store the chat in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle User Input
if prompt := st.chat_input("E.g., What are the safety features?"):
    
    # Show the user's question on screen
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5. Connect to your FastAPI Backend
    with st.spinner("Scanning the manual..."):
        try:
            # Send the request to localhost:8000
            response = requests.post(
                "http://localhost:8000/api/chat",
                json={"question": prompt}
            )
            
            if response.status_code == 200:
                # Extract the answer from the JSON response
                ai_answer = response.json().get("answer", "Error reading response.")
            else:
                ai_answer = f"API Error {response.status_code}: {response.text}"
                
        except requests.exceptions.ConnectionError:
            ai_answer = "⚠️ Error: Could not connect to the backend. Is your FastAPI server running on port 8000?"

    # 6. Show the AI's answer on screen
    with st.chat_message("assistant"):
        st.markdown(ai_answer)
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})