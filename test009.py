import os
import tempfile
from llama_index import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.llms import load_llm
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import Settings
import streamlit as st

# Replace the file path with your actual file path
file_path = r"C:\Users\juan1\Documents\Machine Learning\ChatbotML\test1\example.pdf"

st.header("Document Processing")

if os.path.exists(file_path):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))
            
            # Copy the file to the temporary directory
            with open(file_path, "rb") as src_file:
                with open(temp_file_path, "wb") as dest_file:
                    dest_file.write(src_file.read())
            
            st.write("Indexing your document...")

            loader = SimpleDirectoryReader(
                input_dir=temp_dir,
                required_exts=[".pdf"],
                recursive=True
            )

            docs = loader.load_data()

            # Setup LLM & embedding model
            llm = load_llm()
            st.write("Loading embedding model...")
            embed_model = HuggingFaceEmbedding(model_name="nomic-ai/modernbert-embed-base", trust_remote_code=True, cache_folder='./hf_cache')
            st.write("Embedding model loaded!")

            # Creating an index over loaded data
            Settings.embed_model = embed_model
            index = VectorStoreIndex.from_documents(docs, show_progress=True)

            # Create the query engine
            Settings.llm = llm
            query_engine = index.as_query_engine(streaming=True)

            # Customize prompt template
            qa_prompt_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
            )

            # Inform the user that the file is processed
            st.success("Ready to Chat!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("The specified file path does not exist. Please check the file path and try again.")
