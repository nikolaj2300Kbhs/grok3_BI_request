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

# xAI API endpoint (hypothetical, replace with actual endpoint from xAI docs)
XAI_API_URL = "https://api.x.ai/v1/completions"  # Update with correct Grok 3 endpoint

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
        Simulate the score by analyzing trends in past member reactions, product variety, retail value, brand reputation, category ratings, and surprise value. Return a satisfaction score on a 1-5 scale, with exactly two decimal places (e.g., 4.23). Return only the numerical score.
        """

        headers = {
            "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "grok-3",  # Use "grok-3-mini" for faster responses if needed
            "prompt": prompt,
            "temperature": 0,  # Deterministic output
            "max_tokens": 50,
            "seed": 42  # For reproducibility
        }

        scores = []
        for _ in range(5):  # Run 5 times and average
            response = requests.post(XAI_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            score = result.get("choices", [{}])[0].get("text", "").strip()
            logger.info(f"Run response: {score}")

            if not score:
                logger.error("Model returned an empty response")
                raise ValueError("Model returned an empty response")

            try:
                score_float = float(score)
                if not (1 <= score_float <= 5):
                    raise ValueError("Score out of range")
                scores.append(score_float)
            except ValueError as e:
                logger.error(f"Invalid score format: {score}, error: {str(e)}")
                raise ValueError(f"Invalid score format: {score}")

        if not scores:
            logger.error("No valid scores collected")
            raise ValueError("No valid scores collected")

        # Calculate average score
        avg_score = sum(scores) / len(scores)
        final_score = f"{avg_score:.2f}"
        logger.info(f"Averaged score: {final_score}")
        return final_score

    except Exception as e:
        logger.error(f"Error in box score simulation: {str(e)}")
        raise Exception(f"Error in box score simulation: {str(e)}")

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    if not os.getenv('XAI_API_KEY'):
        raise ValueError("XAI_API_KEY environment variable is not set")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
