import os
import json
from google import genai
from google.genai import types

class BusinessAnalyst:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_insights(self, forecast_summary: dict, metrics: dict, shap_importance: dict, executive_context: str) -> dict:
        """
        Synthesizes ML outputs into a strategic briefing using Gemini.
        Enforces a strict JSON response schema.
        """
        prompt = f"""
        You are a highly analytical Chief Data Officer. Analyze the following time-series forecasting model outputs and provide a strategic business briefing.
        Tell it like it is. Do not sugar-coat the risks.
        
        Context: {executive_context}
        
        Model Evaluation Metrics:
        {json.dumps(metrics, indent=2)}
        
        Forecast Bounds Summary:
        {json.dumps(forecast_summary, indent=2)}
        
        Global Feature Drivers (SHAP Importance):
        {json.dumps(shap_importance, indent=2)}
        
        Respond ONLY with a valid JSON object strictly matching this schema:
        {{
            "executive_summary": "A concise, forward-thinking 2-3 sentence overview.",
            "confidence_assessment": "Honest assessment of model reliability based on MAE, RMSE, and MAPE.",
            "key_drivers": ["driver 1", "driver 2"],
            "risks": ["risk 1", "risk 2"],
            "opportunities": ["opportunity 1", "opportunity 2"],
            "recommended_actions": ["action 1", "action 2"]
        }}
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2 
                )
            )
            return json.loads(response.text)
        except Exception as e:
            raise RuntimeError(f"GenAI extraction failed: {str(e)}")