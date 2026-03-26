#  Voice-Enabled Cursor AI

A voice-controlled AI assistant capable of executing coding and file system operations using natural language. The system leverages LangGraph to perform multi-step planning and tool-based execution.

---

##  Features

*  Voice-based interaction using SpeechRecognition
*  Autonomous multi-step planning using LangGraph
*  Tool-based execution (create, read, write, append, delete files)
*  Run Python scripts directly from commands
*  Safe deletion with user confirmation
*  Stateful interaction handling (pending actions)

---

##  Architecture

The system follows a planner–executor workflow:

User Input → Planner → Executor → Chatbot → Tools → Execution Loop → End

* **Planner**: Breaks tasks into steps
* **Executor**: Executes steps sequentially
* **Chatbot**: Decides tool usage
* **Tools**: Perform actual operations

---

##  Tech Stack

* Python
* LangGraph
* LangChain
* OpenAI GPT-4o API
* SpeechRecognition
* MongoDB (checkpoint memory)
* Docker (MongoDB container)
* subprocess

---

##  Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/saurabhkmr1092/AgentAI.git
cd AgentAI
```

---

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Add environment variables

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

### 5. Run MongoDB (Docker)

```
docker run -d -p 27018:27017 --name mongodb mongo
```

---

### 6. Run the project

```
python main.py
```

---

##  Example Commands

* "Create a Python file for stack implementation"
* "Write code for stack operations"
* "Run the file"
* "Delete stack file"

---

##  Safety Design

Destructive operations like file deletion are handled outside the AI agent with explicit user confirmation to prevent unintended actions.

---

##  Future Improvements

*  Text-to-Speech (voice output)
*  Web search integration
*  FastAPI project generation
*  Improved file understanding

---

##  Author

Saurabh Kumar
