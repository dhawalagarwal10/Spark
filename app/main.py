from fastapi import FastAPI, Request
from app.spark_core import generate_response

app = FastAPI(title="Spark: The Intuity of Being")

@app.post("/spark")
async def spark_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    response = await generate_response(user_input)
    return {"response": response}