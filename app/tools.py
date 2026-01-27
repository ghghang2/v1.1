# app/tools.py
import json
from typing import Callable, Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    schema: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        # Build a minimal JSON‑schema from the function signature.
        # For this demo we hard‑code the schema, but you can introspect
        # annotations for a more general solution.
        if self.name == "get_stock_price":
            self.schema = {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock symbol, e.g. AAPL"},
                    },
                    "required": ["ticker"],
                },
            }
        else:
            raise NotImplementedError("Schema auto‑generation not implemented")

def get_stock_price(ticker: str) -> str:
    data = {"AAPL": 24, "GOOGL": 178.20, "NVDA": 580.12}
    price = data.get(ticker, "unknown")
    return json.dumps({"ticker": ticker, "price": price})

# Register the tool
TOOLS: List[Tool] = [
    Tool(
        name="get_stock_price",
        description="Get the current stock price for a ticker",
        func=get_stock_price,
    )
]

def get_tools() -> List[Dict]:
    """
    Return the list of tool definitions formatted for the OpenAI API.
    Each element has the required `"type": "function"` wrapper.
    """
    api_tools = []
    for t in TOOLS:
        api_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema["parameters"],
                },
            }
        )
    return api_tools