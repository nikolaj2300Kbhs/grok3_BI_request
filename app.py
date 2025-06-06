from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# xAI API endpoint
XAI_API_URL = "https://api.x.ai/v1/completions"  # Adjust if xAI provides a different endpoint

def call_grok3(prompt, max_tokens=500, temperature=0.7):
    """Helper function to call Grok 3 API with a prompt."""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-3",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": 42
        }
        response = requests.post(XAI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        response_text = result.get("choices", [{}])[0].get("text", "").strip()
        
        # Clean response (remove <|separator|> and trailing text)
        cleaned_text = response_text.split('<|separator|>')[0].strip()
        if not cleaned_text:
            logger.error("Grok 3 returned an empty response")
            raise ValueError("Grok 3 returned an empty response")
        
        logger.info(f"Grok 3 response: {cleaned_text}")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error calling Grok 3: {str(e)}")
        raise Exception(f"Error calling Grok 3: {str(e)}")

def predict_box_score(historical_data, future_box_info):
    """Simulate a 1-5 satisfaction score for a future box using Grok 3."""
    try:
        prompt = f"""
        You are a Goodiebox satisfaction expert simulating a member satisfaction score for a future subscription box. Use this data context:
        **Data Explanation**:
        - Historical Data: Past boxes with details like:
          - Box SKU: Unique box identifier (e.g., DK-2504-CLA-2L).
          - Products: Number of items, listed as Product SKUs.
          - Total Retail Value: Sum of product retail prices in €.
          - Unique Categories: Number of distinct product categories (e.g., skincare, makeup).
          - Full-size/Premium: Counts of full-size items and those > €20.
          - Total Weight: Sum of product weights in grams.
          - Avg Brand/Category Ratings: Average ratings (out of 5).
          - Historical Score: Past average box rating (out of 5, e.g., 4.23).
        - Future Box Info: Details of a new box (same format, no historical score yet).
        **Inputs**:
        Historical Data: {historical_data}
        Future Box Info: {future_box_info}
        Simulate the score by analyzing trends in past member reactions, product variety, retail value, brand reputation, category ratings, and surprise value. Return a satisfaction score on a 1-5 scale, with exactly two decimal places (e.g., 4.23). Return only the numerical score (e.g., 4.23).
        """
        
        scores = []
        for _ in range(5):  # Run 5 times and average
            score_text = call_grok3(prompt, max_tokens=50, temperature=0)
            try:
                score_float = float(score_text)
                if not (1 <= score_float <= 5):
                    raise ValueError("Score out of range")
                scores.append(score_float)
            except ValueError as e:
                logger.error(f"Invalid score format: {score_text}, error: {str(e)}")
                raise ValueError(f"Invalid score format: {score_text}")
        
        if not scores:
            logger.error("No valid scores collected")
            raise ValueError("No valid scores collected")
        
        avg_score = sum(scores) / len(scores)
        final_score = f"{avg_score:.2f}"
        logger.info(f"Averaged score: {final_score}")
        return final_score
    except Exception as e:
        logger.error(f"Error in box score simulation: {str(e)}")
        raise Exception(f"Error in box score simulation: {str(e)}")

def analyze_bi(data_context, query):
    """Analyze BI data using Grok 3 and return insights."""
    try:
        prompt = f"""
        You are a BI expert for Goodiebox, a Danish subscription business selling beauty product boxes across 10+ European markets. Analyze the provided data to answer the query. Use clear, concise language suitable for business stakeholders. Return numerical results (if applicable) and a brief explanation.

        **Data Context**:
        - Data Source: Pirate Funnel data (daily metrics per market).
        - Metrics: Intake (new members, reactivations), CAC (cost per acquisition, €), ad spend (€), sales (daily actuals).
        - Markets: Denmark, Germany, Sweden, Norway, Poland, Finland, Netherlands, Belgium, Switzerland, Austria.
        - Time Period: January to June 2025.
        - Example Data: {data_context[:1000]}... (truncated for brevity; use trends and patterns).
        - Notes: Belgium price change on March 10, 2025 (base price from €12.48 to €11.98, delivery from €0 to €1.99).

        **Query**:
        {query}

        **Instructions**:
        - For numerical results (e.g., averages, deltas), return in a JSON-like format: {"results": {"metric": value, ...}}.
        - Provide a concise explanation (2-3 sentences) of the results or trends.
        - If the query is open-ended, focus on key drivers (e.g., price perception, ad spend, market dynamics).
        - If data is insufficient, note limitations and provide a reasonable estimate or suggestion.
        """
        
        response_text = call_grok3(prompt, max_tokens=1000, temperature=0.7)
        return response_text
    except Exception as e:
        logger.error(f"Error in BI analysis: {str(e)}")
        raise Exception(f"Error in BI analysis: {str(e)}")

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    """Endpoint for simulating future box scores."""
    try:
        data = request.get_json()
        if not data or 'future_box_info' not in data:
            logger.error("Missing future box info")
            return jsonify({"error": "Missing future box info"}), 400
        historical_data = data.get('historical_data', 'No historical data provided')
        future_box_info = data['future_box_info']
        score = predict_box_score(historical_data, future_box_info)
        return jsonify({'predicted_box_score': score})
    except Exception as e:
        logger.error(f"Error in /predict_box_score endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_bi', methods=['POST'])
def bi_analysis():
    """Endpoint for BI analysis using Grok 3."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            logger.error("Missing query")
            return jsonify({"error": "Missing query"}), 400
        data_context = data.get('data_context', 'No data context provided')
        query = data['query']
        analysis = analyze_bi(data_context, query)
        return jsonify({'analysis': analysis})
    except Exception as e:
        logger.error(f"Error in /analyze_bi endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    if not os.getenv('XAI_API_KEY'):
        raise ValueError("XAI_API_KEY environment variable is not set")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
