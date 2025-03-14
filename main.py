import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner, WebSearchTool

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")

# Define WebSearchTool for real-time web scraping
web_search_tool = WebSearchTool(
    user_location=None,  # Optional: Customize search results based on location.
    search_context_size="medium",  # Context size for search results.
)

# Define Agents
news_scraper_agent = Agent(
    name="News Scraper Agent",
    instructions="You search the web to find the latest news articles about a given topic. Summarize the main points clearly and concisely.",
    tools=[web_search_tool],
    model="gpt-4o"
)

news_writer_agent = Agent(
    name="News Writer Agent",
    instructions="You write a clear and engaging news article based on the summaries provided by the News Scraper Agent.",
    model="gpt-4o"
)

file_writer_agent = Agent(
    name="File Writer Agent",
    instructions="You save the provided content into a text file with a filename based on the topic.",
    model="gpt-4o"
)

# Define Workflow
async def run_workflow(topic):
    # Step 1: Scrape News
    print(f"Running News Scraper for topic: {topic}...")
    scrape_result = await Runner.run(news_scraper_agent, f"Find and summarize news articles about '{topic}'.")
    summaries = scrape_result.final_output
    print(f"Summaries:\n{summaries}\n")
    print("__________________")

    # Step 2: Write News Article
    print(f"Running News Writer for topic: {topic}...")
    write_result = await Runner.run(news_writer_agent, f"Write a news article about '{topic}' based on these summaries in Markdown format:\n{summaries}")
    article_content = write_result.final_output
    print(f"Article Content:\n{article_content}\n")

    # Step 3: Save to File
    print(f"Running File Writer for topic: {topic}...")
    save_result = await Runner.run(file_writer_agent, f"Save this article to a file named '{topic}_news.md':\n{article_content}")
    
    # Save content locally (for demonstration purposes)
    with open(f"{topic}_news.md", "w") as f:
        f.write(article_content)
    
    print(f"Article saved to '{topic}_news.md'.")

# Main function to execute workflow
async def main():
    topic_input = "OPenai sdk"  # Replace with your desired topic
    await run_workflow(topic=topic_input)

if __name__ == "__main__":
    asyncio.run(main())