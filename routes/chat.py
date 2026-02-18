"""
Chat API routes - Gemini AI assistant endpoint
"""
from flask import Blueprint, request, jsonify
import os
import google.generativeai as genai
from utils import load_litter_df

chat_api = Blueprint('chat_api', __name__, url_prefix='/api')

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


@chat_api.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint that uses Google Gemini AI to answer questions about environmental data
    and make predictions for reef health insights.
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Load environmental dataset for context
        df = load_litter_df()
        
        # Create dataset summary for context
        dataset_summary = f"""
Dataset Information:
- Total beaches: {len(df)}
- Countries: {', '.join(df['country'].unique())}
- Years: 2012-2023 (yearly abundance and survey counts)
- Total litter items recorded: {df['totalLitter'].sum():,.0f}
- Average litter per beach: {df['avgLitter'].mean():.2f}
- Clusters: {', '.join(map(str, df['cluster'].unique()))}

Key columns:
- YYYY_abund: Litter abundance for year YYYY
- YYYY_nbsur: Number of surveys for year YYYY
- totalLitter: Total litter across all years
- avgLitter: Average litter per survey
- litter_slope: Trend slope for linear regression
- litter_intercept: Y-intercept for predictions
- predicted_litter_2025: Predicted litter for 2025
- cluster: Beach grouping based on patterns
- robustGrowthRate: Growth rate of litter over time

Sample data (first 3 beaches):
{df.head(3).to_string()}
"""
        
        # System prompt with dataset context
        system_prompt = """You are Simek, an intelligent assistant for the ReefSpark platform. 
ReefSpark is dedicated to predicting reef bleaching events and providing unified, clean oceanographic data insights to help protect coral reefs.

You help users analyze marine data including beach litter patterns from European beaches (2012-2023), which serves as environmental indicators for reef health.

Your capabilities:
1. Answer questions about environmental trends, patterns, and statistics related to reef health
2. Compare data between beaches and countries to identify environmental stressors
3. Make predictions using the provided slope and intercept values to forecast future conditions
4. Explain data insights and trends that affect coral reef ecosystems
5. Calculate statistics and aggregations to support reef conservation efforts

Our mission: We provide unified, clean data and insights to predict and prevent reef bleaching, addressing the current lack of standardized oceanographic data.

When making predictions:
- Use the formula: predicted_value = litter_slope × year + litter_intercept
- For example, for 2026: predicted_2026 = litter_slope × 2026 + litter_intercept
- Explain your calculations clearly and relate findings to reef health when relevant

Be conversational, helpful, and data-driven. If you need to perform calculations, show your work.
"""
        
        # Call Google Gemini API (using free tier model)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        full_prompt = f"{system_prompt}\n\nDataset Context:\n{dataset_summary}\n\nUser Question: {user_message}"
        
        response = model.generate_content(full_prompt)
        assistant_message = response.text
        
        # Estimate tokens (Gemini doesn't provide exact count in basic API)
        estimated_tokens = len(full_prompt.split()) + len(assistant_message.split())
        
        return jsonify({
            'response': assistant_message,
            'model': 'gemini-2.5-flash',
            'tokens': estimated_tokens
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
