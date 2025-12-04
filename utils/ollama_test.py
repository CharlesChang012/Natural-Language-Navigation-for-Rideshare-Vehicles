import ollama

res = ollama.chat(
    model="qwen3-vl:235b-cloud",
    messages=[
        {
            "role": "user",
            "content": "What is happening in this image do you think?",
            "images": ["/home/sidd/carla_proj/Natural-Language-Navigation-for-Rideshare-Vehicles/utils/ street.jpg"]   # local file paths
        }
    ]
)

print(res["message"]["content"])
