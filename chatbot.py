from vdb import docsearch
import os

from groq import Groq


client = Groq(
    api_key="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"
)


print ("==============================================================================")
while True:

    question = input("Type a question: ")

    r_docs = docsearch.similarity_search(query=question,k=2)

    context = "\n\n".join(doc.page_content for doc in r_docs)

    # print(context)


    system_prompt = "you are a help full assistant"

    chat_completion = client.chat.completions.create(
        messages=[
            {"role":"system", "content": "You are a helpful asistant"},
            {
                "role": "user",
                "content": f"""Answer the following question based on the context provided,If you
                do not know the answer say do not know. DO NOT SAY "Based on the given context"
                
                ## Question: 
                {question}

                ## Context:
                {context}
                """,
            }
        ],
        model="llama3-8b-8192",
    )

    print(chat_completion.choices[0].message.content)

