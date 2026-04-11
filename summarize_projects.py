from ollama import Client

from utils import load_projects


if __name__ == "__main__":

    client = Client(host='http://192.168.2.12:11434')
    
    projects = load_projects("data")

    print(f"There are {len(projects)} projects")

    for i, row in projects.iterrows():

        resp = client.generate(
            model='gpt-oss:20b',
            prompt=f"""Rewrite this project summary into 2 concise sentences.
                        Keep:
                        - main goal
                        - technology or approach
                        - application domain

                        Do not invent information.
                        Maximum 80 words.
                        
                        Project: {row["objective"]}""",
            stream=False
        )

        print(resp["response"])


        if i == 5:
            break

    
        

