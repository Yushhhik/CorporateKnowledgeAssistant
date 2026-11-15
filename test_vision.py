import base64
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

image_path = r"C:\Users\natag\Downloads\CorporateKnowledgeAssistant-main\test_image.jpg"

with open(image_path, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

try:
    print("⏳ Отправляем картинку в LLaVA...")
    response = client.chat.completions.create(
        model="llava",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Что на этой картинке? Опиши подробно."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        max_tokens=300
    )
    print("✅ Модель ответила!")
    print("Ответ:", response.choices[0].message.content)
except Exception as e:
    print("❌ Ошибка:", e)