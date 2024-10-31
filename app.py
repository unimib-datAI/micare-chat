from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import csv 
import os
import time

modelli = ["gemma2:latest","gemma2:27b","phi3:medium","phi3.5:latest","mistral-nemo:latest","llama3:latest","mixtral:8x7b"]
#modelli = ["phi3.5:latest"]
# loading the vectorstore
for modello in modelli:
    print("-------------------------------------------------------------")
    print(f"modello: {modello} -- {modelli.index(modello)+1}/{len(modelli)}")
    vectorstore = Chroma(persist_directory=f"./chroma_db-{modello.replace(':','')}", embedding_function=OllamaEmbeddings(model=modello))

    # loading the Llama3 model
    llm = Ollama(model=modello)

    # using the vectorstore as the retriever
    retriever = vectorstore.as_retriever()

    # formating the docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # loading the QA chain from langchain hub
    rag_prompt = hub.pull("rlm/rag-prompt")

    # creating the QA chainhi
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    with open('./output/splitdsm.csv',encoding="utf-8") as inputFile: 
        
        # Skips the heading 
        # Using next() method 
        heading = next(inputFile) 
        
        # Create reader object by passing the file  
        # object to reader method 
        reader_obj = csv.reader(inputFile,delimiter="§") 
        directory = f"./output/{modello.replace(':','')}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Iterate over each row in the csv file  
        # using reader object 
        i = 1
        startTime = time.time()
        prevTime = time.time() - startTime
        for row in reader_obj: 
            # if time.time() - startTime > 3600 or i > 20:
            #     break
            print(f"modello: {modello} -- Riga {i}/103 -- Tempo elaborazione: {(time.time() - startTime)-prevTime}s Tempo passato: {time.time() - startTime}s -- Tempo mancante: {(((time.time() - startTime)/i)*103)-(time.time() - startTime)}s ({((((time.time() - startTime)/i)*103)-(time.time() - startTime))/3600}h)")
            with open("./log.txt","a",encoding="utf-8") as log:
                log.write(f"modello: {modello} -- Riga {i}/103 -- Tempo elaborazione: {(time.time() - startTime)-prevTime}s Tempo passato: {time.time() - startTime}s -- Tempo mancante: {(((time.time() - startTime)/i)*103)-(time.time() - startTime)}s ({((((time.time() - startTime)/i)*103)-(time.time() - startTime))/3600}h)\n")
            prevTime = time.time() - startTime
            try:
                question = f"based on the icd-11 make a diagnosis for this case: {row[1]}"
                answer = qa_chain.invoke(question)
                answer = answer.replace("\n","")
                with open(directory+"/answers-dsm.csv","a",encoding="utf-8") as outputFile:
                    if i==1:
                        outputFile.write("row§quesion§answer\n")
                    outputFile.write(f"{i}§{question}§{answer}\n")
            except:
                question = "no question"
                with open(directory+"/answers-dsm.csv","a",encoding="utf-8") as outputFile:
                    outputFile.write(f"{i}§{question}§NO ANSWER\n")
            i=i+1


    # running the QA chain in a loop until the user types "exit"
  #  while True:
  #      question = input("Question: ")
  #      if question.lower() == "exit":
  #          break
  #      answer = qa_chain.invoke(question)

  #      print(f"\nAnswer: {answer}\n")



