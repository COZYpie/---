import json
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Union
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
import os
from openai import UnprocessableEntityError
import aiohttp
import time

# Configure logging
logging.basicConfig(
    filename='backend.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(title="Travel Planner Backend")

# Set SSL cert file environment variable
os.environ["SSL_CERT_FILE"] = ""

# Initialize LLM (DeepSeek) with increased timeout
model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="sk-8d71a948abc44172ae9471247099b92b",
    openai_api_base="https://api.deepseek.com/v1",
    max_tokens=1024,
    temperature=0.3,
    timeout=60,  # Increased timeout to 60 seconds
)


# Pydantic models for request validation
class City(BaseModel):
    name: str
    days: int


class PlanRequest(BaseModel):
    mode: str
    userInput: str
    conversation_id: str
    city: str | None = None
    days: int | None = None
    cities: List[City] | None = None


def estimate_tokens(text: str) -> int:
    """Roughly estimate the number of tokens in a text"""
    return len(text) // 4 + len(text.split()) // 2


def truncate_text(text: str, max_tokens: int, suffix: str = "...") -> str:
    """Truncate text to fit within a token limit"""
    if estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    return text[:max_chars - len(suffix)] + suffix


async def init_mcp_client():
    """Initialize MCP client and return tools"""
    async with MultiServerMCPClient(
            {
                "gaode": {
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                }
            }
    ) as client:
        tools = client.get_tools()
        logging.debug(f"MCP client tools: {tools}")
        return tools


async def generate_summary_agent(agent, plan: Dict, user_input: str) -> str:
    """Generate a detailed and comprehensive summary for the travel plan"""
    user_input = truncate_text(user_input, 3000)
    summary_prompt = """
    你是一个专业的旅行顾问，擅长为用户提供详细、连贯且吸引人的旅行计划总结。
    请根据以下详细的旅行计划和用户输入的偏好，生成一个最终的总结性建议，全面整合所有规划细节。
    总结必须包括以下内容：
    - **旅行亮点**：突出每个部分的独特体验（如文化景点的历史意义、美食的特色、住宿的便利性、交通的优化安排等）。
    - **行程安排概述**：描述整体行程的节奏（紧凑或轻松）、每天的主要活动，以及适合的旅行者类型（家庭、情侣、独旅等）。
    - **天气影响**：说明天气对行程的潜在影响及应对建议。
    - **实用建议**：提供具体建议，如最佳旅行时间、穿着推荐、预算规划、当地礼仪或安全注意事项。
    - **用户偏好融入**：明确如何根据用户输入的偏好（如文化、美食、冒险等）定制行程。
    旅行计划：{plan}
    用户偏好：{user_input}
    输出格式为纯文本，结构清晰，使用小标题分段（如“旅行亮点”、“行程安排”等），字数控制在400-600字，确保详细但不过于冗长。语气友好、吸引人，突出小金毛导航的贴心服务。
    """
    messages_summary_agent = [
        SystemMessage(summary_prompt.format(plan=json.dumps(plan, ensure_ascii=False), user_input=user_input)),
        HumanMessage("请生成最终的详细总结性建议")
    ]

    total_tokens = sum(estimate_tokens(msg.content) for msg in messages_summary_agent)
    logging.debug(f"Summary agent token count: {total_tokens}")
    if total_tokens > 60000:
        logging.warning(f"Summary token count {total_tokens} exceeds safe limit, truncating user_input")
        user_input = truncate_text(user_input, 1000)
        messages_summary_agent[0] = SystemMessage(
            summary_prompt.format(plan=json.dumps(plan, ensure_ascii=False), user_input=user_input))
        total_tokens = sum(estimate_tokens(msg.content) for msg in messages_summary_agent)
        logging.debug(f"After truncation, summary token count: {total_tokens}")

    for attempt in range(3):
        try:
            response = await agent.ainvoke({"messages": messages_summary_agent})
            summary = response['messages'][-1].content
            if not isinstance(summary, str):
                logging.warning(f"Summary is not a string: {summary}, converting to string")
                summary = str(summary)
            logging.debug(f"Generated summary: {summary}")
            return summary
        except Exception as e:
            logging.error(f"Summary agent attempt {attempt + 1} failed: {str(e)}", exc_info=True)
            if attempt == 2:
                return f"总结生成失败：{str(e)}"
            await asyncio.sleep(2 ** attempt)  # Exponential backoff


async def single_city(agent, city_name: str, days: int, user_input: str) -> Dict:
    """Handle single-city itinerary planning"""
    logging.info(f"Planning itinerary for {city_name} for {days} days with input: {user_input}")

    user_input = truncate_text(user_input, 3000)  # Stricter initial limit

    # Task decomposition
    messages_core = [
        SystemMessage(
            """将用户输入的旅游需求拆分为景区、住宿、餐饮、出行四个方面的详细要求。
            每个字段必须是一个字符串，不能是列表或对象。
            输出仅为一个有效的JSON字符串，格式如下：
            {"景区": "字符串描述", "住宿": "字符串描述", "餐饮": "字符串描述", "出行": "字符串描述"}"""
        ),
        HumanMessage(user_input),
    ]

    logging.debug(f"Task decomposition messages: {messages_core}")
    total_tokens = sum(estimate_tokens(msg.content) for msg in messages_core)
    logging.debug(f"Task decomposition token count: {total_tokens}")

    for attempt in range(3):
        try:
            response_core = await agent.ainvoke({"messages": messages_core})
            core_content = response_core['messages'][-1].content
            logging.debug(f"Raw task decomposition response: {core_content}")
            break
        except Exception as e:
            logging.error(f"Task decomposition attempt {attempt + 1} failed: {str(e)}", exc_info=True)
            if attempt == 2:
                yield {"error": f"Task decomposition failed after retries: {str(e)}"}
                return
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

    try:
        messages_core_dict = json.loads(core_content)
        logging.debug(f"Parsed task decomposition dict: {messages_core_dict}")
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in task decomposition: {core_content}, Error: {e}")
        messages_core[0] = SystemMessage(
            """将用户输入的旅游需求拆分为景区、住宿、餐饮、出行四个方面的详细要求。
            每个字段必须是一个字符串，不能是列表或对象。
            必须输出一个有效的JSON字符串，格式如下：
            {"景区": "字符串描述", "住宿": "字符串描述", "餐饮": "字符串描述", "出行": "字符串描述"}。
            不要包含任何非JSON内容，如解释性文本或前缀。"""
        )
        logging.debug(f"Retrying task decomposition with messages: {messages_core}")
        for attempt in range(3):
            try:
                response_core = await agent.ainvoke({"messages": messages_core})
                core_content = response_core['messages'][-1].content
                messages_core_dict = json.loads(core_content)
                logging.debug(f"Retry task decomposition response: {core_content}")
                logging.debug(f"Retry parsed task decomposition dict: {messages_core_dict}")
                break
            except json.JSONDecodeError as e:
                logging.error(f"Retry JSON decode failed: {core_content}, Error: {e}")
                if attempt == 2:
                    yield {"error": f"Task decomposition failed after retries: JSON decode error: {str(e)}"}
                    return
            except Exception as e:
                logging.error(f"Retry task decomposition attempt {attempt + 1} failed: {str(e)}", exc_info=True)
                if attempt == 2:
                    yield {"error": f"Task decomposition failed after retries: {str(e)}"}
                    return
                await asyncio.sleep(2 ** attempt)

    view = messages_core_dict.get('景区', '')
    accommodation = messages_core_dict.get('住宿', '')
    food = messages_core_dict.get('餐饮', '')
    traffic = messages_core_dict.get('出行', '')
    if not isinstance(food, str):
        logging.warning(f"Food is not a string: {food}, converting to string")
        food = str(food)
    logging.debug(f"Task decomposition result: {messages_core_dict}")

    plan = {"city": city_name, "days": days}

    # Weather query
    messages_weather = [
        SystemMessage(
            "使用工具查询指定城市的当前及未来数日天气情况，并以简洁的文本形式返回。"
        ),
        HumanMessage(f"查询{city_name}的天气")
    ]
    logging.debug(f"Weather query messages: {messages_weather}")
    for attempt in range(3):
        try:
            response_weather = await agent.ainvoke({"messages": messages_weather})
            weather_plan = response_weather['messages'][-1].content
            if not isinstance(weather_plan, str):
                logging.warning(f"Weather plan is not a string: {weather_plan}, converting to string")
                weather_plan = str(weather_plan)
            weather_plan = truncate_text(weather_plan, 3000)
            plan["weather"] = weather_plan
            yield {"partial": {"weather": weather_plan}}
            logging.debug(f"Yielded partial weather: {weather_plan}")
            break
        except Exception as e:
            logging.error(f"Weather query attempt {attempt + 1} failed: {str(e)}", exc_info=True)
            if attempt == 2:
                yield {"error": f"Weather query failed after retries: {str(e)}"}
                return
            await asyncio.sleep(2 ** attempt)

    # Scenic spots planning
    view = truncate_text(view, 1500)
    messages_view = [
        SystemMessage(
            f"根据天气情况和旅游攻略意见{view}，推荐适合的游玩景点。"
        ),
        HumanMessage(user_input)
    ]
    total_tokens = sum(estimate_tokens(msg.content) for msg in messages_view)
    logging.debug(f"Scenic spots messages: {messages_view}")
    logging.debug(f"Scenic spots token count: {total_tokens}")
    for attempt in range(3):
        try:
            response_view = await agent.ainvoke({"messages": messages_view})
            view_plan = response_view['messages'][-1].content
            if not isinstance(view_plan, str):
                logging.warning(f"Scenic spots plan is not a string: {view_plan}, converting to string")
                view_plan = str(view_plan)
            view_plan = truncate_text(view_plan, 3000)
            plan["scenic_spots"] = view_plan
            yield {"partial": {"scenic_spots": view_plan}}
            logging.debug(f"Yielded partial scenic spots: {view_plan}")
            break
        except Exception as e:
            logging.error(f"Scenic spots planning attempt {attempt + 1} failed: {str(e)}", exc_info=True)
            if attempt == 2:
                yield {"error": f"Scenic spots planning failed after retries: {str(e)}"}
                return
            await asyncio.sleep(2 ** attempt)

    # Dining planning
    food = truncate_text(food, 1500)
    user_input_truncated = truncate_text(user_input, 2000)  # Stricter limit
    messages_food = [
        SystemMessage(
            "推荐交通方便、口碑好的美食地点。"
        ),
        HumanMessage(
            f"{user_input_truncated} 结合美食评价，推荐美食地点。攻略意见：{food}"
        )
    ]
    total_tokens = sum(estimate_tokens(msg.content) for msg in messages_food)
    logging.debug(f"Dining messages: {messages_food}")
    logging.debug(f"Dining inputs - user_input: {user_input_truncated}, food: {food}, estimated tokens: {total_tokens}")
    if total_tokens > 60000:
        logging.warning(f"Dining token count {total_tokens} exceeds safe limit, further truncating")
        user_input_truncated = truncate_text(user_input_truncated, 1000)
        food = truncate_text(food, 1000)
        messages_food[1] = HumanMessage(
            f"{user_input_truncated} 结合美食评价，推荐美食地点。攻略意见：{food}"
        )
        total_tokens = sum(estimate_tokens(msg.content) for msg in messages_food)
        logging.debug(f"After truncation, dining token count: {total_tokens}")

    for attempt in range(3):
        try:
            response_food = await agent.ainvoke({"messages": messages_food})
            food_plan = response_food['messages'][-1].content
            if not isinstance(food_plan, str):
                logging.warning(f"Dining plan is not a string: {food_plan}, converting to string")
                food_plan = str(food_plan)
            food_plan = truncate_text(food_plan, 3000)
            plan["dining"] = food_plan
            yield {"partial": {"dining": food_plan}}
            logging.debug(f"Yielded partial dining: {food_plan}")
            break
        except UnprocessableEntityError as e:
            logging.error(f"Dining planning attempt {attempt + 1} failed (UnprocessableEntityError): {str(e)}",
                          exc_info=True)
            if attempt == 2:
                yield {"error": f"Dining planning failed after retries: {str(e)}"}
                return
        except Exception as e:
            logging.error(f"Dining planning attempt {attempt + 1} failed: {str(e)}", exc_info=True)
            if attempt == 2:
                yield {"error": f"Dining planning failed after retries: {str(e)}"}
                return
            await asyncio.sleep(2 ** attempt)

    # Accommodation planning
    accommodation = truncate_text(accommodation, 1500)
    view_plan_truncated = truncate_text(view_plan, 2000)
    messages_accommodation = [
        SystemMessage(
            "推荐交通方便、靠近景区的住宿地点。"
        ),
        HumanMessage(
            f"{user_input_truncated} 参考景点规划：{view_plan_truncated}，推荐酒店住宿。攻略意见：{accommodation}"
        )
    ]
    total_tokens = sum(estimate_tokens(msg.content) for msg in messages_accommodation)
    logging.debug(f"Accommodation messages: {messages_accommodation}")
    logging.debug(
        f"Accommodation inputs - user_input: {user_input_truncated}, view_plan: {view_plan_truncated}, accommodation: {accommodation}, estimated tokens: {total_tokens}")
    if total_tokens > 60000:
        logging.warning(f"Accommodation token count {total_tokens} exceeds safe limit, further truncating")
        user_input_truncated = truncate_text(user_input_truncated, 1000)
        view_plan_truncated = truncate_text(view_plan_truncated, 1000)
        accommodation = truncate_text(accommodation, 1000)
        messages_accommodation[1] = HumanMessage(
            f"{user_input_truncated} 参考景点规划：{view_plan_truncated}，推荐酒店住宿。攻略意见：{accommodation}"
        )
        total_tokens = sum(estimate_tokens(msg.content) for msg in messages_accommodation)
        logging.debug(f"After truncation, accommodation token count: {total_tokens}")

    for attempt in range(3):
        try:
            response_accommodation = await agent.ainvoke({"messages": messages_accommodation})
            accommodation_plan = response_accommodation['messages'][-1].content
            if not isinstance(accommodation_plan, str):
                logging.warning(f"Accommodation plan is not a string: {accommodation_plan}, converting to string")
                accommodation_plan = str(accommodation_plan)
            accommodation_plan = truncate_text(accommodation_plan, 3000)
            plan["accommodation"] = accommodation_plan
            yield {"partial": {"accommodation": accommodation_plan}}
            logging.debug(f"Yielded partial accommodation: {accommodation_plan}")
            break
        except Exception as e:
            logging.error(f"Accommodation planning attempt {attempt + 1} failed: {str(e)}", exc_info=True)
            if attempt == 2:
                yield {"error": f"Accommodation planning failed after retries: {str(e)}"}
                return
            await asyncio.sleep(2 ** attempt)

    # Transportation planning
    traffic = truncate_text(traffic, 1500)
    accommodation_plan_truncated = truncate_text(accommodation_plan, 2000)
    messages_traffic = [
        SystemMessage(
            "提供详细的出行方案。"
        ),
        HumanMessage(
            f"{user_input_truncated} 参考景点规划：{view_plan_truncated}，住宿：{accommodation_plan_truncated}，提供出行路线规划。攻略意见：{traffic}"
        )
    ]
    total_tokens = sum(estimate_tokens(msg.content) for msg in messages_traffic)
    logging.debug(f"Transportation messages: {messages_traffic}")
    logging.debug(
        f"Transportation inputs - user_input: {user_input_truncated}, view_plan: {view_plan_truncated}, accommodation: {accommodation_plan_truncated}, traffic: {traffic}, estimated tokens: {total_tokens}")
    if total_tokens > 60000:
        logging.warning(f"Transportation token count {total_tokens} exceeds safe limit, further truncating")
        user_input_truncated = truncate_text(user_input_truncated, 1000)
        view_plan_truncated = truncate_text(view_plan_truncated, 1000)
        accommodation_plan_truncated = truncate_text(accommodation_plan_truncated, 1000)
        traffic = truncate_text(traffic, 1000)
        messages_traffic[1] = HumanMessage(
            f"{user_input_truncated} 参考景点规划：{view_plan_truncated}，住宿：{accommodation_plan_truncated}，提供出行路线规划。攻略意见：{traffic}"
        )
        total_tokens = sum(estimate_tokens(msg.content) for msg in messages_traffic)
        logging.debug(f"After truncation, transportation token count: {total_tokens}")

    for attempt in range(3):
        try:
            response_traffic = await agent.ainvoke({"messages": messages_traffic})
            traffic_plan = response_traffic['messages'][-1].content
            if not isinstance(traffic_plan, str):
                logging.warning(f"Transportation plan is not a string: {traffic_plan}, converting to string")
                traffic_plan = str(traffic_plan)
            traffic_plan = truncate_text(traffic_plan, 3000)
            plan["transportation"] = traffic_plan
            yield {"partial": {"transportation": traffic_plan}}
            logging.debug(f"Yielded partial transportation: {traffic_plan}")
            break
        except Exception as e:
            logging.error(f"Transportation planning attempt {attempt + 1} failed: {str(e)}", exc_info=True)
            if attempt == 2:
                yield {"error": f"Transportation planning failed after retries: {str(e)}"}
                return
            await asyncio.sleep(2 ** attempt)

    # Detailed itinerary summary
    food_plan_truncated = truncate_text(food_plan, 2000)
    traffic_plan_truncated = truncate_text(traffic_plan, 2000)
    messages_summary = [
        SystemMessage(
            f"综合景区安排{view_plan_truncated}，餐饮安排{food_plan_truncated}，住宿安排{accommodation_plan_truncated}，出行安排{traffic_plan_truncated}，生成详细单项总结。"
        ),
        HumanMessage(user_input_truncated)
    ]
    total_tokens = sum(estimate_tokens(msg.content) for msg in messages_summary)
    logging.debug(f"Detailed itinerary summary messages: {messages_summary}")
    logging.debug(
        f"Summary inputs - view_plan: {view_plan_truncated}, food_plan: {food_plan_truncated}, accommodation: {accommodation_plan_truncated}, traffic: {traffic_plan_truncated}, estimated tokens: {total_tokens}")
    if total_tokens > 60000:
        logging.warning(f"Summary token count {total_tokens} exceeds safe limit, further truncating")
        user_input_truncated = truncate_text(user_input_truncated, 1000)
        view_plan_truncated = truncate_text(view_plan_truncated, 1000)
        food_plan_truncated = truncate_text(food_plan_truncated, 1000)
        accommodation_plan_truncated = truncate_text(accommodation_plan_truncated, 1000)
        traffic_plan_truncated = truncate_text(traffic_plan_truncated, 1000)
        messages_summary[0] = SystemMessage(
            f"综合景区安排{view_plan_truncated}，餐饮安排{food_plan_truncated}，住宿安排{accommodation_plan_truncated}，出行安排{traffic_plan_truncated}，生成详细单项总结。"
        )
        messages_summary[1] = HumanMessage(user_input_truncated)
        total_tokens = sum(estimate_tokens(msg.content) for msg in messages_summary)
        logging.debug(f"After truncation, summary token count: {total_tokens}")

    for attempt in range(3):
        try:
            response_summary = await agent.ainvoke({"messages": messages_summary})
            detailed_summary = response_summary['messages'][-1].content
            if not isinstance(detailed_summary, str):
                logging.warning(f"Detailed summary is not a string: {detailed_summary}, converting to string")
                detailed_summary = str(detailed_summary)
            detailed_summary = truncate_text(detailed_summary, 3000)
            plan["summary"] = detailed_summary
            yield {"partial": {"summary": detailed_summary}}
            logging.debug(f"Yielded partial detailed summary: {detailed_summary}")
            break
        except Exception as e:
            logging.error(f"Detailed summary planning attempt {attempt + 1} failed: {str(e)}", exc_info=True)
            if attempt == 2:
                yield {"error": f"Detailed summary planning failed after retries: {str(e)}"}
                return
            await asyncio.sleep(2 ** attempt)

    # Final comprehensive summary
    try:
        final_summary = await generate_summary_agent(agent, plan, user_input)
        plan["final_summary"] = final_summary
        yield {"partial": {"final_summary": final_summary}}
        logging.debug(f"Yielded final summary: {final_summary}")
    except Exception as e:
        logging.error(f"Final summary generation failed: {str(e)}", exc_info=True)
        yield {"error": f"Final summary generation failed: {str(e)}"}
        return

    # Ensure complete plan is yielded
    try:
        yield {"complete": plan}
        logging.debug(f"Yielded complete plan: {plan}")
        logging.info(f"Completed itinerary for {city_name}")
    except Exception as e:
        logging.error(f"Failed to yield complete plan: {str(e)}", exc_info=True)
        yield {"error": f"Failed to yield complete plan: {str(e)}"}
        return


async def multi_city(agent, cities: List[Dict], user_input: str) -> List[Dict]:
    """Handle multi-city itinerary planning"""
    logging.info(f"Planning multi-city itinerary for cities: {[c['name'] for c in cities]}")
    complete_plans = []

    for city in cities:
        logging.debug(f"Processing city: {city['name']}")
        async for partial in single_city(agent, city["name"], city["days"], user_input):
            yield partial
            logging.debug(f"Yielded partial for {city['name']}: {partial}")
            if "complete" in partial:
                complete_plans.append(partial["complete"])
            elif "error" in partial:
                logging.error(f"Error in city {city['name']}: {partial['error']}")
                yield {"error": f"Error in city {city['name']}: {partial['error']}"}
                return

    try:
        yield {"complete": complete_plans}
        logging.debug(f"Yielded complete multi-city plan: {complete_plans}")
        logging.info("Completed multi-city itinerary")
    except Exception as e:
        logging.error(f"Failed to yield complete multi-city plan: {str(e)}", exc_info=True)
        yield {"error": f"Failed to yield complete multi-city plan: {str(e)}"}


async def stream_plan(request: PlanRequest):
    """Stream itinerary planning results"""
    logging.info(f"Received request: {request.dict()}")

    # Validate request
    if request.mode not in ["single", "multi"]:
        logging.error(f"Invalid mode: {request.mode}")
        yield json.dumps({"error": "Invalid mode. Use 'single' or 'multi'"}, ensure_ascii=False) + "\n"
        return

    if request.mode == "single" and (not request.city or not request.days):
        logging.error("Missing city or days for single mode")
        yield json.dumps({"error": "Single mode requires city and days"}, ensure_ascii=False) + "\n"
        return

    if request.mode == "multi" and (not request.cities or len(request.cities) == 0):
        logging.error("Missing cities for multi mode")
        yield json.dumps({"error": "Multi mode requires at least one city"}, ensure_ascii=False) + "\n"
        return

    try:
        async with MultiServerMCPClient(
                {
                    "gaode": {
                        "url": "http://localhost:8000/sse",
                        "transport": "sse",
                    }
                }
        ) as client:
            agent = create_react_agent(model, client.get_tools())

            if request.mode == "single":
                async for result in single_city(agent, request.city, request.days, request.userInput):
                    if not isinstance(result, dict):
                        logging.error(f"Unexpected result type: {type(result)}")
                        yield json.dumps({"error": f"Unexpected result type: {type(result)}"},
                                         ensure_ascii=False) + "\n"
                        return
                    yield json.dumps(result, ensure_ascii=False) + "\n"
            else:
                async for result in multi_city(agent, [city.dict() for city in request.cities], request.userInput):
                    if not isinstance(result, dict):
                        logging.error(f"Unexpected result type: {type(result)}")
                        yield json.dumps({"error": f"Unexpected result type: {type(result)}"},
                                         ensure_ascii=False) + "\n"
                        return
                    yield json.dumps(result, ensure_ascii=False) + "\n"

    except Exception as e:
        logging.error(f"Error during planning: {str(e)}", exc_info=True)
        yield json.dumps({"error": f"Planning failed: {str(e)}"}, ensure_ascii=False) + "\n"


@app.post("/api/plan/stream")
async def plan_stream(request: PlanRequest):
    """FastAPI endpoint for streaming travel plans"""
    return StreamingResponse(
        stream_plan(request),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)