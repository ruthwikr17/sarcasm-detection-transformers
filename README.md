Sarcasm Detection using Transformer Models

A complete end-to-end NLP system that detects sarcasm in short-form text using fine-tuned Transformer models.
The project includes model training pipelines, evaluation scripts, and a modern Flask-based web dashboard for real-time sarcasm prediction and model comparison.


Project Overview

Sarcasm often expresses meaning opposite to literal text, making it difficult for traditional sentiment analysis systems to interpret correctly.
This project builds a context-aware sarcasm detection system by fine-tuning Transformer-based language models to classify tweets as:
	•	Sarcastic
	•	Not Sarcastic

In addition to training high-performing models, the project provides an interactive web dashboard where users can enter any text, select a model, and instantly receive sarcasm predictions with confidence scores.


Key Features
	•	Fine-tuned BERT and RoBERTa models for sarcasm classification
	•	Real-time Flask dashboard for single-text prediction
	•	Model comparison mode to evaluate BERT vs RoBERTa on the same input
	•	Confidence visualization for each prediction
	•	Clean, dark-themed UI for demonstration and deployment
	•	Modular training, evaluation, and inference scripts


Technologies Used

Programming Language
	•	Python 3.9+

Deep Learning & NLP
	•	PyTorch
	•	Hugging Face Transformers

Models
	•	BERT (bert-base-uncased)
	•	RoBERTa (roberta-base)

Data Processing
	•	Pandas, NumPy

Evaluation
	•	Scikit-learn (Accuracy, Precision, Recall, F1-score)

Web Deployment
	•	Flask
	•	HTML, CSS, JavaScript


Dataset

The models are trained and evaluated on labeled tweet datasets containing sarcastic and non-sarcastic samples.

Binary Classification Labels:
	•	1 → Sarcastic
	•	0 → Not Sarcastic

Data preprocessing includes:
	•	Cleaning and normalization
	•	Class balancing
	•	Tokenization using model-specific tokenizers
	•	Train-test split to avoid data leakage


System Workflow

Input Tweet
     ↓
Tokenizer (BERT / RoBERTa)
     ↓
Fine-tuned Transformer Model
     ↓
Softmax Classification Layer
     ↓
Prediction + Confidence Score
     ↓
Web Dashboard Output



Model
Accuracy
Precision
Recall
F1-Score
RoBERTa
0.885
0.871
0.899
0.885
BERT
0.862
0.845
0.881
0.863



Project Structure

Sarcasm-Detection-RoBERTa/
│
├── app.py                     
├── templates/                
├── static/                
│
├── Bert.py                   
├── RoBERTa.py          
├── TestBert.py                
├── TestRoberta.py   
├── metrics.py                
├── Cleanup.py             
│
├── Data/                    
├── requirements.txt
└── README.md


Running the Web Dashboard

Install Dependencies
pip install -r requirements.txt

Run Flask App
python app.py

Open in Browser
http://127.0.0.1:5000/


Dashboard Capabilities
	•	Enter any tweet or sentence
	•	Select BERT or RoBERTa model
	•	Detect sarcasm in real-time
	•	View prediction label and confidence
	•	Compare both models on same input

Example Predictions

Input:
“Oh great, another meeting that could’ve been an email. Fantastic.”

Output:
Prediction: Sarcastic
Confidence: 0.9997


Future Enhancements
	•	Multilingual sarcasm detection
	•	Larger and more diverse datasets
	•	Context-aware conversation-based sarcasm detection

Author

Ruthvik Reddy
AI / ML Developer
GitHub: https://github.com/ruthwikr17
