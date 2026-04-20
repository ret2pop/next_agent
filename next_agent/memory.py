import os
import time
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import email
from email import policy
from .vars import EMBEDDING_MODEL

class CodebaseRAG:
    def __init__(self, repo_path: str, db_path: str, index_name: str = "monorepo_index", valid_exts = (
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".md",
            ".txt",
            ".json",
            ".org",
            ".nix",
        ),
                 email_store = False
                 ):
        self.repo_path = repo_path
        self.db_path = db_path
        self.index_name = index_name
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = None
        self.valid_exts = valid_exts
        self.email_store = email_store
        
        os.makedirs(self.db_path, exist_ok=True)
        self.load_existing_db()

    def search(self, query: str, k: int = 3) -> str:
        if not self.vector_store:
            return "❌ Error: Monorepo index not built. Run /rebuild."
        
        results = self.vector_store.similarity_search(query, k=k)
        if not results:
            return "No relevant code found."
        
        context_parts = []
        for doc in results:
            path = os.path.abspath(doc.metadata.get('source'))
            context_parts.append(f"SOURCE_FILE_PATH: {path}\nCONTENT:\n{doc.page_content}\n---")
        return "\n".join(context_parts)

    def load_existing_db(self):
        index_path = os.path.join(self.db_path, f"{self.index_name}.faiss")
        if os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    self.index_name,
                    allow_dangerous_deserialization=True,
                )
                print("📁 Existing vector database loaded.")
            except Exception as e:
                print(f"⚠️ Could not load existing DB: {e}")

    def rebuild_index(self):
        start_time = time.time()
        print(f"\n🔍 [1/3] Scanning directory: {self.repo_path}...")

        documents = []
        file_count = 0

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [
                d
                for d in dirs
                if d
                not in {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}
            ]

            for file in files:
                if (self.valid_exts is None) or (os.path.splitext(file)[1].lower() in self.valid_exts):
                    file_path = os.path.join(root, file)
                    try:
                        if self.email_store:
                            with open(file_path, "rb") as f:
                                msg = email.message_from_binary_file(f, policy=policy.default)
                                body = ""
                                
                                for part in msg.walk():
                                    if part.get_content_type() == "text/plain":
                                        body += part.get_content()
                                
                                if not body.strip():
                                    body = msg.get("Subject", "No body text.")

                                if body.strip():
                                    documents.append(
                                        Document(
                                            page_content=body,
                                            metadata={
                                                "source": file_path,
                                                "from": msg.get("From", "Unknown"),
                                                "subject": msg.get("Subject", "No Subject")
                                            },
                                        )
                                    )
                                    
                        else:
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                                if content.strip():
                                    documents.append(
                                        Document(
                                            page_content=content,
                                            metadata={"source": file_path},
                                        )
                                    )
                                    
                        file_count += 1
                        if file_count % 20 == 0:
                            print(f"  • Processed {file_count} files...", end="\r")
                            
                    except Exception:
                        continue

        print(f"\n✅ Found {len(documents)} valid files.")

        print("✂️ [2/3] Chunking text into smaller pieces...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks.")

        print(f"🧬 [3/3] Generating Embeddings via Ollama ({EMBEDDING_MODEL})...")
        print("⚠️  This is the slow part. Please wait, do not exit.")

        try:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.save_local(self.db_path, self.index_name)

            duration = time.time() - start_time
            print(f"🏁 Finished! Database built in {duration:.1f} seconds.")
        except Exception as e:
            print(f"\n❌ Embedding Error: {e}")
            print("Check if Ollama is running and you have your embedding model pulled.")
