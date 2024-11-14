from model import LanguageModel

def main():
    model = LanguageModel()  

    print("加载训练好的模型...")

    while True:

        question = input("你好！请问你有什么问题？输入 'exit' 退出程序。\n")
        

        if question.lower() == 'exit':  

            print("退出程序...")

            break

        response = model.get_unique_answer(question)  

        print(f"模型的回答：{response}\n")

if __name__ == "__main__":
    main()
