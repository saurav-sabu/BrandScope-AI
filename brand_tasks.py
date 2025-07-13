from crewai import Task


class BrandTask():

    def __validate_inputs(self, brand_name, competitors):
        if not brand_name or not competitors:
            raise ValueError("Brand name and competitor list must be provided")
        return True

    def search_task(self, agent, brand_name, competitors):
        self.__validate_inputs(brand_name, competitors)
        return Task(
            description=f"""
            You are the Search Agent assigned to collect up-to-date online data about the brand **{brand_name}** and its competitors: {', '.join([c['name'] for c in competitors])}.

            Use search engines and APIs to:
            - Fetch 1 latest high-quality news or blog entries per brand.
            - Prioritize reputable, timely, and relevant sources.
            - Summarize each source clearly for use by downstream agents.
            """,
                        expected_output="""
            [
            {
                "brand": "Nike",
                "title": "Nike launches eco-friendly shoes",
                "url": "https://...",
                "source": "CNN",
                "publishedAt": "2025-07-03",
                "summary": "Nike introduced a new line of sustainable footwear..."
            },
            ...
            ]
            """,
                        agent=agent
                    )
    

    def sentiment_task(self, agent):
        return Task(
            description="""
You are the Sentiment Analyst Agent.

Your input is a collection of article summaries about brands. For each brand:
- Count articles with positive, neutral, and negative tone.
- Assign a sentiment score from -1 (negative) to +1 (positive).

Highlight subtle emotional cues and ignore sarcastic/misleading signals.
""",
            expected_output="""
[
  {
    "brand": "Nike",
    "positive": 6,
    "neutral": 2,
    "negative": 2,
    "avg_sentiment_score": 0.60
  },
  ...
]
""",
            agent=agent
        )

    def finance_task(self, agent, brand_name, competitors):
        self.__validate_inputs(brand_name, competitors)
        return Task(
            description=f"""
        You are the Financial Analyst.

        Analyze financial performance for **{brand_name}** and its competitors ({', '.join([c['ticker'] for c in competitors])}) using YFinance API.

        For each brand:
        - Retrieve current price, 7-day trend (%), and volatility.
        - Identify any investor-impacting news or anomalies.
        """,
                    expected_output="""
        [
        {
            "ticker": "NKE",
            "company": "Nike",
            "current_price": 98.40,
            "change_7d": 1.25,
            "volatility": "Moderate"
        },
        ...
        ]
        """,
                    agent=agent
                )

    def comparison_task(self, agent, brand_name, competitors):
        self.__validate_inputs(brand_name, competitors)
        return Task(
            description=f"""
        You are the Competitive Intelligence Analyst.

        Compare the overall brand performance of **{brand_name}** with competitors: {', '.join([c['name'] for c in competitors])}.

        Use:
        - Sentiment analysis results
        - Search visibility / article count
        - Financial change % over 7 days

        Generate a side-by-side comparison and declare which brand leads overall and why.
        """,
                    expected_output="""
        | Brand   | Mentions | Sentiment Score | 7-Day Trend (%) | Verdict |
        |---------|----------|-----------------|------------------|---------|
        | Nike    | 12       | 0.60            | +1.25%           | 🟢 Best performance |
        | Adidas  | 8        | 0.45            | +0.75%           | 🟡 Moderate |
        | Puma    | 4        | -0.10           | -0.80%           | 🔴 Weak |

        Summary:
        - Nike is ahead in sentiment, visibility, and financial strength.
        """,
                    agent=agent
                )

    def report_task(self, agent, brand_name):
        return Task(
            description=f"""
        You are the Executive Reporting Specialist.

        Consolidate outputs from all other agents and prepare a final report for **{brand_name}**.

        The report should:
        - Summarize sentiment trends
        - Highlight financial performance
        - Compare with competitors
        - Offer 2-3 strategic recommendations

        Format should be business-friendly and visually clear.
        """,
                    expected_output=f"""

        Return the final output in **valid GitHub-flavored Markdown**, clearly divided into the following 5 sections. Use headings (##), bullet points, and a Markdown table where needed.

        Example format:
        
        # 📊 Brand Monitoring Report: {brand_name}

        ## 1. Public Sentiment Overview
        - Positive: ...
        - Negative: ...
        - Score: ...

        ## 2. Financial Summary
        - Current price: ...
        - Weekly change: ...

        ## 3. Competitor Comparison
        - Table with scores and ranking

        ## 4. Key Insights
        - Summary bullets

        ## 5. Recommendations
        - Actionable next steps for brand team

        Generate in proper markdown format
        """
        ,
                    agent=agent
                )

