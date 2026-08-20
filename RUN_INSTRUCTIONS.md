# How to Run the MedAssist AI Project

Follow these steps to launch the Django server and access the web application on your local machine.

## 1. Open your Terminal

Navigate to the root directory of the project where this file is located:
```bash
cd "/Users/sanyamgehlot/Desktop/Main/My Projects/Medicial_Knowledge_Assistant_using_RAG"
```

## 2. Activate the Virtual Environment

Before running the server, you need to activate the Python virtual environment to ensure all dependencies are correctly loaded:
```bash
source venv/bin/activate
```
*(You should see `(venv)` appear at the beginning of your terminal prompt.)*

## 3. Start the Django Development Server

Run the following command to start the server:
```bash
python manage.py runserver
```

## 4. Access the Application

Once the server says "Starting development server at http://127.0.0.1:8000/", you can open your web browser and navigate to:

- 💬 **Patient/Doctor Chat Interface**: [http://127.0.0.1:8000/chat/](http://127.0.0.1:8000/chat/)
- 🛡️ **Model Control / Admin Panel**: [http://127.0.0.1:8000/model-control/](http://127.0.0.1:8000/model-control/)

## 5. Stopping the Server

When you are done, simply go to your terminal window where the server is running and press:
`Control + C`

*(You can then type `deactivate` to exit the virtual environment if you wish).*

---

### ⚠️ Note on Ollama (AI Model)
For the AI chat to actually respond to queries, ensure that the **Ollama** application is running in the background on your Mac, and that the `med-llama` model is available. If Ollama is not running, the web interface will still load, but the AI won't be able to generate responses.
