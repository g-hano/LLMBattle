# LLMBattle
*Will add better README soon*

Make sure you have Ollama installed, after that run those commands on the terminal
```bash
ollama pull llama2
ollama pull mistral
ollama pull nomic-embed-text
```

Create virtual env
```bash
python -m venv venv
```

Clone the repo
```bash
git clone https://github.com/g-hano/LLMBattle.git
venv\scripts\activate
cd LLMBattle
pip install -r requirements.txt
```

Run the Gradio app
```bash
python app.py
```
