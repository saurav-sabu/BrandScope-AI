from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from brand_agents import BrandAgent
from brand_tasks import BrandTask
from crewai import Crew
import os
from functools import lru_cache
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Brand Monitor API",
    description="AI-powered brand monitoring and competitive analysis using CrewAI",
    version="1.0.0"
)

# Enable CORS for all origins (for development/demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request model for brand analysis
class CompetitorInput(BaseModel):
    name: str = Field(
        ...,
        example="Adidas",
        description="Competitor brand name"
    )
    ticker: str = Field(
        ...,
        example="ADDYY",
        description="Stock ticker symbol"
    )

class BrandAnalysisRequest(BaseModel):
    brand_name: str = Field(
        ...,
        example="Nike",
        description="Brand name to analyze"
    )
    competitors: List[CompetitorInput] = Field(
        ...,
        example=[
            {"name": "Adidas", "ticker": "ADDYY"},
            {"name": "Puma", "ticker": "PUMSY"}
        ],
        description="List of competitors with their ticker symbols"
    )

# Response model for brand analysis
class BrandAnalysisResponse(BaseModel):
    status: str
    message: str
    report: Optional[str] = None
    error: Optional[str] = None

# Settings class to load API keys from environment
class Settings:
    def __init__(self):
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        self.BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")

# Cached settings loader
@lru_cache()
def get_settings():
    return Settings()

# Main class to orchestrate brand monitoring using CrewAI
class BrandCrew:
    """
    Handles the orchestration of brand monitoring using CrewAI agents and tasks.
    """
    def __init__(self, brand_name, competitors):
        self.brand_name = brand_name
        self.competitors = competitors

    def run(self):
        """
        Runs the brand monitoring process by initializing agents, tasks, and Crew.
        Returns the generated brand report or None if an error occurs.
        """
        try:
            # Initialize agent and task classes
            agents = BrandAgent()
            tasks = BrandTask()

            # Create agents for different roles
            search_agent = agents.search_agent_brand()
            sentiment_agent = agents.sentiment_analyst_agent()
            finance_agent = agents.finance_analyst_agent()
            comparison_agent = agents.comparison_analyst_agent()
            report_agent = agents.report_agent()

            # Define tasks for each agent
            search_task = tasks.search_task(
                search_agent,
                self.brand_name,
                self.competitors
            )
            sentiment_task = tasks.sentiment_task(sentiment_agent)
            finance_task = tasks.finance_task(
                finance_agent,
                self.brand_name,
                self.competitors
            )
            comparison_task = tasks.comparison_task(
                comparison_agent,
                self.brand_name,
                self.competitors
            )
            report_task = tasks.report_task(
                report_agent,
                self.brand_name
            )

            # Create Crew with agents and tasks
            crew = Crew(
                agents=[search_agent, sentiment_agent, finance_agent, comparison_agent, report_agent],
                tasks=[search_task, sentiment_task, finance_task, comparison_task, report_task],
                verbose=True
            )

            # Run the Crew to generate the brand report
            result = crew.kickoff()
            return result.raw

        except Exception as e:
            # Raise HTTPException for FastAPI error handling
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

# Root endpoint for API health/info
@app.get("/")
async def root():
    return {
        "message": "Welcome to Brand Monitor API",
        "docs": "/docs",
        "redoc_url": "/redoc"
    }

# Main endpoint to analyze brand
@app.post("/api/v1/analyze-brand", response_model=BrandAnalysisResponse)
async def analyze_brand(request: BrandAnalysisRequest):
    # Validate competitors list
    if not request.competitors or len(request.competitors) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one competitor must be provided"
        )
    
    # Convert competitors to the format expected by tasks
    competitors_list = [{"name": comp.name, "ticker": comp.ticker} for comp in request.competitors]

    try:
        # Initialize brand crew and generate report
        brand_crew = BrandCrew(
            request.brand_name,
            competitors_list
        )
        report = brand_crew.run()

        # Return successful response
        return BrandAnalysisResponse(
            status="SUCCESS",
            message="Brand analysis completed successfully",
            report=report
        )
    
    except Exception as e:
        # Return error response
        return BrandAnalysisResponse(
            status="ERROR",
            message="Failed to generate brand analysis",
            error=str(e)
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Run the app with Uvicorn if executed as main script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)