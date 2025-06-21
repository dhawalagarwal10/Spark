from app.core_loop import run_core_loop

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    reply = run_core_loop(user_input)
    print(f"Spark: {reply}\n")