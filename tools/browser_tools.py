import json
import requests
import streamlit as st
import logging
from crewai.tools import BaseTool
from unstructured.partition.html import partition_html
from crewai import Task, Agent, LLM
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteInput(BaseModel):
    """
    Defines the schema for the website scraping input.
    """
    website: str = Field(..., description="The website URL to scrape")

class BrowserTools(BaseTool):
    """
    Tool for scraping and summarizing website content.
    """
    name: str = "Scrape the website content"
    description: str = "Useful to scrape and summarize a website content"
    args_schema: type[BaseModel] = WebsiteInput

    def _run(self, website: str) -> str:
        """
        Scrapes the content of a website and summarizes it using an LLM agent.
        """
        try:
            logger.info(f"Starting website scraping for: {website}")

            # Prepare API endpoint and headers for browserless.io
            api_key = st.secrets["BROWSERLESS_API_KEY"]
            url = f"https://chrome.browserless.io/content?token={api_key}"
            payload = json.dumps({"url": website})
            headers = {
                "Cache-Control": "no-cache",
                "Content-Type": "application/json"
            }

            logger.info("Sending POST request to browserless.io API")
            response = requests.post(url, headers=headers, data=payload)

            if response.status_code != 200:
                logger.error(f"Search API request failed. Status Code: {response.status_code}")
                return f"Error: Search API request failed. Status Code: {response.status_code}"

            logger.info("Partitioning HTML content")
            elements = partition_html(text=response.text)
            content = "\n\n".join(str(el) for el in elements)

            logger.info("Splitting content into manageable chunks")
            chunk_size = 8000
            content_chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            summaries = []

            logger.info("Initializing LLM model")
            llm = LLM(model="gemini/gemini-2.0-flash")

            for idx, chunk in enumerate(content_chunks):
                logger.info(f"Processing chunk {idx+1}/{len(content_chunks)}")
                agent = Agent(
                    role="Principal Researcher",
                    goal="Conduct in-depth research to gather accurate, relevant, and insightful information that supports strategic decision-making.",
                    backstory=(
                        "You are a highly analytical and detail-driven Principal Researcher with years of experience synthesizing complex information into actionable insights. "
                        "Known for your methodical approach and critical thinking, you specialize in uncovering valuable patterns, trends, and data-driven stories. "
                        "Your work enables teams to make informed choices across domains such as travel, business, technology, or policy. "
                        "You prioritize clarity, accuracy, and relevance in every report you produce."
                    ),
                    allow_delegation=False,
                    llm=llm
                )

                task = Task(
                    description=(
                        "You are tasked with performing high-quality background research on the assigned topic. "
                        "This may include collecting data from reliable sources, summarizing key insights, comparing options, and identifying notable trends or considerations.\n\n"
                        f"**Topic**: {chunk}\n\n"
                        "Your goal is to:\n"
                        "- Analyze credible, up-to-date sources.\n"
                        "- Structure your findings clearly and concisely.\n"
                        "- Ensure all data supports the decision or planning process that follows.\n\n"
                        "Use a formal, well-organized tone and include references if relevant. Present your output as a research summary with headings, bullet points, and clear structure."
                    ),
                    agent=agent
                )

                logger.info(f"Executing summarization task for chunk {idx+1}")
                summary = task.execute()
                summaries.append(summary)

            logger.info("Combining all summaries")
            return "\n\n".join(summaries)

        except Exception as e:
            logger.error(f"Error while processing the website: {str(e)}")
            return f"Error while processing the website: {str(e)}"