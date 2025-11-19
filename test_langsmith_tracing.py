"""Test LangSmith tracing with LangChain/LangGraph operations."""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 70)
print("LangSmith Tracing Test")
print("=" * 70)
print(f"LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
print(f"LANGSMITH_API_KEY: {os.getenv('LANGSMITH_API_KEY', '')[:25]}...")
print("=" * 70)
print()

# Test 1: Simple LangChain LLM call
print("Test 1: Simple LangChain LLM Call")
print("-" * 70)
try:
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    response = llm.invoke([HumanMessage(content="What is the capital of France? Answer in one sentence.")])

    print(f"Response: {response.content}")
    print("[OK] LangChain LLM call completed - check LangSmith dashboard for trace")
    print()
except Exception as e:
    print(f"Error: {e}")
    print()

# Test 2: LangChain Agent with Tool
print("Test 2: LangChain Agent with Tool")
print("-" * 70)
try:
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"The weather in {city} is sunny and 72°F!"

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather],
        system_prompt="You are a helpful assistant that provides weather information.",
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in New York?"}]})

    # Extract the final response
    final_message = result["messages"][-1]
    print(f"Agent Response: {final_message.content}")
    print("[OK] LangChain Agent execution completed - check LangSmith dashboard for trace")
    print()
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
    print()

# Test 3: LangGraph StateGraph (if available)
print("Test 3: LangGraph StateGraph")
print("-" * 70)
try:
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    class GraphState(TypedDict):
        messages: list

    def process_node(state: GraphState):
        """Process messages."""
        return {"messages": state["messages"] + [{"role": "assistant", "content": "Processed!"}]}

    # Build graph
    workflow = StateGraph(GraphState)
    workflow.add_node("process", process_node)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)

    app = workflow.compile()

    result = app.invoke({"messages": [{"role": "user", "content": "Test message"}]})

    print(f"Graph Result: {result['messages'][-1]['content']}")
    print("[OK] LangGraph execution completed - check LangSmith dashboard for trace")
    print()
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
    print()

print("=" * 70)
print("All tests completed!")
print("=" * 70)
print("Check your LangSmith dashboard for traces:")
print(f"  Project: {os.getenv('LANGSMITH_PROJECT')}")
print(f"  URL: https://smith.langchain.com/o/{os.getenv('LANGSMITH_ORG_ID', 'your-org')}/projects")
print("=" * 70)
