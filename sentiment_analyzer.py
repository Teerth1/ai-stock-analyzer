# Sentiment Analyzer using FinBERT

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FinBertAnalyzer:
    """FinBERT sentiment analyzer for financial news."""
    
    def __init__(self):
        print("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        print("FinBERT model loaded!")
    
    def analyze(self, text):
        """Analyze sentiment of text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        probs = predictions[0].tolist()
        compound_score = probs[2] - probs[0]  # positive - negative
        
        return {
            'score': compound_score,
            'positive': probs[2],
            'neutral': probs[1],
            'negative': probs[0]
        }


def analyze_sentiment_finbert(text, finbert_analyzer):
    """Helper function to analyze sentiment."""
    result = finbert_analyzer.analyze(text)
    return result['score']
