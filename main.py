from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os
import logging

# Set up logging to help debug issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# OPENROUTE_API_KEY = os.getenv("OPENROUTER_API_KEY")
# if not OPENROUTE_API_KEY:
#     raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

# model = ChatOpenAI(
#     base_url="https://api.openrouter.com/v1",
#     api_key=OPENROUTE_API_KEY,
#     model_name="deepseek/deepseek-chat-3.5",
#     temperature=0.1,
# )

# Validate environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

if not FIRECRAWL_API_KEY:
    logger.warning("FIRECRAWL_API_KEY environment variable is not set.")

# Initialize the model with better error handling
try:
    model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model_name="meta-llama/llama-4-maverick:free",
        temperature=0.1,
        timeout=30,  # Add timeout
        max_retries=3,  # Add retry logic
    )
except Exception as e:
    logger.error(f"Failed to initialize ChatOpenAI model: {e}")
    raise

server_params = StdioServerParameters(
    command = 'npx',
    env = {
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    },
    args = ["firecrawl-mcp"]
)

async def main():
    async with stdio_client(server_params) as (read,write):
        async with ClientSession (read, write) as session:
            await session.initialize()
            tools = await  load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to help the user."
                },
            ]

            print("Available tools -", *[tool.name for tool in tools])
            print("-" * 60)

            while True:
                user_input = input("\nYou: ")
                if user_input == "quit":
                    print("Exiting...")
                    break

                messages.append({"role": "user", "content": user_input[:175000]})

                try:
                    agent_response = await agent.ainvoke({"messages": messages})

                    ai_message = agent_response["message"][-1].content
                    print("\nAgent:", ai_message)
                except Exception as e:
                    print("Error:", e)
if __name__ == "__main__":
    asyncio.run(main())