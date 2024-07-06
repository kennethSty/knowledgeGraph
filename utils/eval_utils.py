from langchain_chroma import Chroma
import chromadb
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv('../config/keys.env', override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_langchain_chroma(persist_dir="/Users/Kenneth/PycharmProjects/knowledgeGraph/data/04_eval/chroma_store"):
    embed_fn = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    client = chromadb.PersistentClient(path=persist_dir)
    langchain_chroma = Chroma(
        client=client,
        collection_name="mesh_embeddings",
        embedding_function=embed_fn,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return langchain_chroma


def get_eval_chain():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    chain = node_evaluation_prompt | llm
    return chain


# Node Evaluation Prompt
node_eval_system_text = (
"# Anweisungen für die Identifizierung von verwandten Begriffen und Synonymen\n"
"## Überblick\n"
"Du erhälst einen Ausdruck, sowie eine Liste von vier weiteren Begriffen. Deine Aufgabe ist es zu entscheiden, ob der gegebene Ausdruck zu mindestens einem der vier weiteren Begriffe eine starke Verwandtschaft aufweist.\n"
"Eine starke Verwandtschaft liegt unter anderem dann vor, wenn der Ausdruck für mindestens einen der vier Begriffe z.B. a) ein Synonym  b) eine speziellere Formulierung  c) eine allgemeinere Formulierung oder d) eine Beschreibung desselben Konzepts ist"
"Liegt eine starke Verwandtschaft vor ist 'True' auszugeben. Liegt keine starke Verwandtschaft zu mindestens einem der vier Begriffe vor ist 'False' auszugeben."
"Bei den Begriffen und dem Ausdruck handelt es sich um medizinische Fachbegriffe. \n"

"## Beispiel 1: \n"
"Der zu prüfende Ausdruck: 'Plötzlicher Herztod' \n"
"Die vier möglicherweise verwandten Begriffe: ['herztod', 'erschöpfung', 'herzmassage', 'herzmuskel']\n"
"Der korrekte Output lautet: 'True', denn  'plötzlicher Herztod' ist eine spezifischere Beschreibung von Herztod. \n"
"## Beispiel 2:\n"
"Der zu prüfende Ausdruck: 'Geistige Behinderung' \n"
"Die vier möglicherweise verwandten Begriffe: ['Schwere Geistige Behinderung, Teilweise Auf Dem Niveau Eines Babys', 'Behinderung', 'Geistige Eingeschränktheit', 'Gehbehinderung'] \n"
"Der korrekte Ouput lautet: 'True', denn 'Geistige Behinderung' ist eine Umschreibung für 'Geistige Eingeschränktheit \n"
"## Beispiel 3:\n"
"Der zu prüfende Ausdruck: 'lissenzephalien mit mutationen im arx-gen'\n"
"Die vier möglicherweise verwandten Begriffe: ['x-chromosomale lissenzephalie', 'x-chromosomal gebundene lissenzephalie', 'lissenzephalie']"
"Der korrekte Ouput lautet: 'True', denn der Ausdruck 'lissenzephalien mit mutationen im arx-gen' ist nur eine spezielle Formulierung des allgemeineren Begriffs 'Lissenzphalie', der in der Liste enthalten ist."
"Gebe ausschließlich **'True'** aus, wenn der gegebene Ausdruck zu einem der vier anderen stark verwandt ist. Ansonsten gebe **'False'** aus."
"Halte Dich strikt an die Regeln. Nichteinhaltung führt zur Beendigung."
)

node_evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            node_eval_system_text,
        ),
        (
            "human",
            (
                "Tipp: Stelle sicher, dass Du ausschließlich mit 'True' oder 'False' antwortest."
                "Gebe keinerlei Erklärungen oder Erläuterungen aus. \n"
                "Achte nicht auf Groß- und Kleinschreibung der Begriffe. \n"
                "Der zu prüfende Ausdruck: {input_term} \n"
                "Die vier möglicherweise verwandten Begriffe lauten: {related_terms}"
            ),
        ),
    ]
)






