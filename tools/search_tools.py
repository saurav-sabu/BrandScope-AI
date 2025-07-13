import json
import requests
import logging
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from a .env file
load_dotenv()

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the schema for the search query input
class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query to look up")

# Define the search tool class
class SearchTools(BaseTool):
    name: str = "Search the Internet"
    description: str = "Useful to search the internet about the given topic and return relevant results"
    args_schema: type[BaseModel] = SearchQuery

    # Main method to run the search
    def _run(self, query: str) -> str:
        try:
            logger.info(f"Starting search for query: {query}")
            top_results_to_return = 4  # Number of top results to return
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query})
            headers = {
                'X-API-KEY': st.secrets["SERPER_API_KEY"],  # API key from Streamlit secrets
                'Content-Type': 'application/json'
            }

            logger.debug(f"Sending POST request to {url} with payload: {payload}")
            # Send POST request to the search API
            response = requests.request("POST", url, headers=headers, data=payload)
            logger.info(f"Received response with status code: {response.status_code}")

            # Check if the response status is not OK
            if response.status_code != 200:
                logger.error(f"Search API request failed with status code: {response.status_code}")
                return f"Error: Search API request failed"

            # Parse the JSON response
            data = response.json()
            logger.debug(f"Response JSON: {data}")

            # Check if 'organic' results are present in the response
            if "organic" not in data:
                logger.warning("No 'organic' results found in response.")
                return "No results found or API Error Occurred"

            results = data["organic"]
            formatted_results = []

            # Format each result for output
            for result in results[:top_results_to_return]:
                try:
                    formatted_result = "\n".join(
                        [
                            f"Title: {result.get('title', 'N/A')}",
                            f"Link: {result.get('link', 'N/A')}",
                            f"Snippet: {result.get('snippet', 'N/A')}"
                        ]
                    )
                    formatted_results.append(formatted_result)
                    logger.debug(f"Formatted result: {formatted_result}")
                except Exception as e:
                    logger.error(f"Error formatting result: {e}")
                    continue

            # Return the formatted results if available
            if formatted_results:
                logger.info(f"Returning {len(formatted_results)} formatted results.")
                return "\n".join(formatted_results)
            else:
                logger.warning("No valid result found after formatting.")
                return "No valid result found"

        except Exception as e:
            # Log and return any exception that occurs during the search
            logger.exception(f"Error during search: {str(e)}")
            return f"Error during search: {str(e)}"
