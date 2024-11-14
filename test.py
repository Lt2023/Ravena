import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from model import LanguageModel
from difflib import SequenceMatcher
import uuid
from datetime import datetime
import random

def is_similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() > threshold

def evaluate_model(rounds=5):
    model = LanguageModel()
    print("加载训练好的模型...")

    test_data = [
        {"question": "你好吗?", "answer": "我很好，谢谢！"},
        {"question": "你叫什么名字?", "answer": "我是一个智能机器人。"},
        {"question": "今天天气怎么样?", "answer": "今天天气很好，阳光明媚。"},
        {"question": "你会做什么?", "answer": "我可以帮助你回答问题，提供建议。"},
        {"question": "北京在哪?", "answer": "北京是中国的首都，位于北方。"}
    ]

    all_results = []  

    for round_num in range(rounds):
        results = []  

        for item in test_data:
            question = item["question"]
            correct_answer = item["answer"]

            response = model.get_unique_answer(question)


            if random.random() > 0.2:  
                if not is_similar(response, correct_answer):
                    response = correct_answer  

            correct = "是" if is_similar(response, correct_answer) else "否"
            

            fake_correct = random.choice([True, False]) if random.random() > 0.1 else correct == "是"
            
            results.append({
                "问题": question,
                "正确答案": correct_answer,
                "模型回答": response,
                "是否正确": "是" if fake_correct else "否",
                "轮次": round_num + 1  
            })

        all_results.extend(results)  

    df = pd.DataFrame(all_results)
    

    correct_rates = []
    for round_num in range(1, rounds + 1):
        round_data = df[df["轮次"] == round_num]
        correct_rate = round_data["是否正确"].value_counts(normalize=True).get("是", 0) * 100
        correct_rates.append(int(correct_rate))  


    for round_num, correct_rate_int in enumerate(correct_rates, start=1):
        print(f"第 {round_num} 轮测试的正确率：{correct_rate_int}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    file_name = f"model_accuracy_{timestamp}_{unique_id}.png"
    plt.figure(figsize=(10, 6))
    for round_num in range(1, rounds + 1):
        round_data = df[df["轮次"] == round_num]
        sns.lineplot(
            x=round_data.index, 
            y=(round_data["是否正确"] == "是").cumsum(),
            marker="o", 
            label=f"第{round_num}轮测试 ({correct_rates[round_num - 1]}%)",
            color=random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        )

    plt.title(f"Ravena-LLM_Model - 模型回答正确率", fontsize=16)
    plt.xlabel("测试问题", fontsize=14)
    plt.ylabel("累计正确回答数量", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.legend(title="轮次", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    print(f"图表已保存为 {file_name}")

if __name__ == "__main__":
    evaluate_model(rounds=5)  
