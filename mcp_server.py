from mcp.server.fastmcp import FastMCP
import httpx

# Initialize the MCP server
mcp = FastMCP("AdaptiveSLA_Manager")

API_URL = "http://127.0.0.1:8000"

@mcp.tool()
def check_system_metrics() -> str:
    """
    Fetch the current CPU usage and prediction latency from the prediction API.
    Use this to monitor for persistent SLA violations (e.g., latency > 100ms).
    """
    try:
        response = httpx.get(f"{API_URL}/metrics")
        return str(response.json())
    except Exception as e:
        return f"Error connecting to API: {str(e)}"

@mcp.tool()
def switch_active_model(model_name: str) -> str:
    """
    Switch the active machine learning model.
    Available models: 'logistic', 'random_forest', 'xgboost'.
    Use this tool when the SLA is violated or when conditions stabilize.
    """
    try:
        response = httpx.post(
            f"{API_URL}/switch_model", 
            json={"model_name": model_name}
        )
        return str(response.json())
    except Exception as e:
        return f"Error switching model: {str(e)}"

if __name__ == "__main__":
    # Run the server using standard input/output for agent communication
    mcp.run()
