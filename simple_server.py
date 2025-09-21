from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Simple server working"}

@app.get("/api/v1/ai/scalping-analysis")
def scalping():
    return {"scalping_opportunities": [], "analysis_method": "AI Scalping Analysis"}

@app.get("/api/v1/ai/options-strategies")
def options():
    return {"options_opportunities": [], "analysis_method": "AI Options Strategy Analysis"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)