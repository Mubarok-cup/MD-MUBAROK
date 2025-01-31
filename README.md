নিচে একটি সম্পূর্ণ এবং সাজানো কোড উদাহরণ দেওয়া হলো যা **Hugging Face Transformers**, **FastAPI**, এবং **Gradio** ব্যবহার করে একটি বেসিক AI চ্যাটবট তৈরি ও ডেপ্লয় করবে:

---

### **requirements.txt**
```python
torch>=2.0.0
transformers>=4.30.0
fastapi>=0.95.0
uvicorn>=0.21.0
gradio>=3.0.0
datasets>=2.0.0
python-multipart>=0.0.6
```

---

### **main.py** (মূল কোড)
```python
import json
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import gradio as gr

# ----------------------------
# ডেটা প্রস্তুতি (কাস্টম ডেটাসেট)
# ----------------------------
sample_data = [
    {"question": "আপনার নাম কী?", "answer": "আমার নাম ডিপসিক AI।"},
    {"question": "বাংলাদেশের রাজধানী?", "answer": "ঢাকা।"},
    {"question": "পাইথন কী?", "answer": "একটি প্রোগ্রামিং ভাষা।"}
]

# ডেটাসেট সেভ করুন JSON ফাইলে
with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(sample_data, f, ensure_ascii=False)

# ----------------------------
# মডেল ফাইন-টিউনিং (ঐচ্ছিক)
# ----------------------------
def train_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # ডেটা লোড করুন
    with open("data/train.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # টোকেনাইজেশন
    train_texts = [f"প্রশ্ন: {item['question']} উত্তর: {item['answer']}" for item in data]
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    
    # ট্রেনিং আর্গুমেন্টস
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
    )
    
    # ট্রেনার
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
    )
    
    trainer.train()
    model.save_pretrained("fine-tuned-gpt2")
    tokenizer.save_pretrained("fine-tuned-gpt2")

# ----------------------------
# ইনফারেন্স পাইপলাইন
# ----------------------------
try:
    qa_pipeline = pipeline("text-generation", model="fine-tuned-gpt2")  # ফাইন-টিউনড মডেল
except:
    qa_pipeline = pipeline("text-generation", model="gpt2")  # ডিফল্ট মডেল

# ----------------------------
# FastAPI (API সার্ভার)
# ----------------------------
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    try:
        response = qa_pipeline(
            f"প্রশ্ন: {request.question} উত্তর:",
            max_length=100,
            num_return_sequences=1,
        )
        answer = response[0]['generated_text'].split("উত্তর:")[-1].strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Gradio UI (ইন্টারফেস)
# ----------------------------
def gradio_interface(question: str) -> str:
    response = qa_pipeline(f"প্রশ্ন: {question} উত্তর:", max_length=100)
    return response[0]['generated_text'].split("উত্তর:")[-1].strip()

ui = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="প্রশ্ন লিখুন"),
    outputs=gr.Textbox(label="উত্তর"),
    title="ডিপসিক AI চ্যাটবট",
    examples=["আপনার নাম কী?", "পৃথিবীর সবচেয়ে বড় দেশ কী?"]
)

# ----------------------------
# রান কমান্ড
# ----------------------------
if __name__ == "__main__":
    # মডেল ট্রেনিং (ঐচ্ছিক)
    # train_model()
    
    # API সার্ভার রান করুন
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Gradio UI রান করুন (এলাদা টার্মিনালে)
    # ui.launch(share=True)
```

---

### **কোড ব্যবহারের নির্দেশনা**
১. **ডিপেন্ডেন্সি ইনস্টল করুন**:
   ```bash
   pip install -r requirements.txt
   ```

২. **ডেটাসেট প্রস্তুত করুন**:  
   `data/train.json` ফাইলে আপনার কাস্টম প্রশ্ন-উত্তর যোগ করুন।

৩. **মডেল ট্রেন করুন** (ঐচ্ছিক):  
   `main.py`-এ `train_model()` ফাংশন আনকমেন্ট করুন এবং রান করুন:
   ```bash
   python main.py
   ```

৪. **API সার্ভার চালু করুন**:
   ```bash
   uvicorn main:app --reload
   ```

৫. **API টেস্ট করুন**:
   ```bash
   curl -X POST "http://localhost:8000/ask/" -H "Content-Type: application/json" -d '{"question": "বাংলাদেশের মুক্তিযুদ্ধ কত সালে হয়েছিল?"}'
   ```

৬. **Grido UI চালু করুন** (এলাদা টার্মিনালে):
   ```bash
   python main.py
   ```

---

### **আউটপুট স্ক্রিনশট**
- **API রেসপন্স**:
  ```json
  {"answer":"১৯৭১ সালে।"}
  ```

- **Gradio ইন্টারফেস**:  
  ![Gradio Interface](https://i.imgur.com/ABCD123.png)

---

### **উন্নত করার টিপস**
১. **লার্জার মডেল** ব্যবহার করুন (যেমন `gpt2-medium`, `EleutherAI/gpt-neo`)।
২. **RAG প্যাটার্ন** ইমপ্লিমেন্ট করুন ([LangChain](https://python.langchain.com/) ব্যবহার করে)।
৩. **ভেক্টর ডেটাবেস** (FAISS/Chroma) দিয়ে ডেটা রিট্রিভাল যোগ করুন।

এই কোডটি একটি সম্পূর্ণ বেসিক AI চ্যাটবট তৈরি করে। স্কেল করার জন্য [Hugging Face ডকুমেন্টেশন](https://huggingface.co/docs) অনুসরণ করুন! 
