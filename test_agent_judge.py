import asyncio
from agent import run_agent, llm_judge

async def main():
    question = "What is cross validation?"
    answer, context = await run_agent(question)

    print("\n--- AGENT ANSWER ---")
    print(answer)
    print("\n--- RETRIEVED CONTEXT (first 200 chars) ---")
    print(context[:200])

    print("\n--- JUDGE SCORES ---")
    judge_result = await llm_judge(question, context, answer)
    print(judge_result)

if __name__ == "__main__":
    asyncio.run(main())