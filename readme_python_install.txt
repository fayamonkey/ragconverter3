Create a new virtual environment:

python -m venv venv

Activate the virtual environment:

.\venv\Scripts\activate

Install the dependencies:

pip install -r requirements.txt

Install the spaCy model:

python -m spacy download en_core_web_sm

Start the app:

streamlit run src/main.py


The app runs at http://localhost:8501/