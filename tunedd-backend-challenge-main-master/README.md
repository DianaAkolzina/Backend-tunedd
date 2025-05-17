# tunedd-backend-challenge

The purpose of this challenge is to assess your Python coding skills in backend and ML development. The task relates closely to our current work on the Tunedd product, and is a good reflection of the kind of work you will be doing, and technologies you will be working with, should you be successful and accept an offer with us. We expect that the core parts of the challenge can be completed in around 2 - 3 hours. Should you wish to, there are some additional items which can be considered for bonus points.

Thank you for your interest in this position with Brainpool AI and devoting your time and effort to the coding challenge. We look forward to reviewing your submission! Good Luck!

# Challenge
Build an API using FastAPI with an endpoint which facilitates a long-running conversation with an LLM grounded by information sourced from a collection of provided documents. In other words, a [RAG chatbot](https://aws.amazon.com/what-is/retrieval-augmented-generation/). The provided documents are a selection of [Arxiv papers in pdf format](./data/ai-agents-arxiv-papers) on AI Agents [discussed in this article](https://deepgram.com/learn/top-arxiv-papers-about-ai-agents). We hope you find this topic as interesting as we do! Put your application code in the [`tunedd_api`](./tunedd_api) directory and any tests in the [`tests`](./tests) directory. Your application should meet the following criteria:

Required:
- Loads the provided data into a vector database prior to application startup for retrieval at query time.
- Provides a FastAPI endpoint to create new conversations.
- Provides a FastAPI endpoint which enables sending a message to an LLM for a specific conversation and returns the response.
- Each conversation has a separate history which the LLM uses to continue that conversation where it left off.
- Retrieves relevant context from the document collection based on the user's message.
- Uses a popular ML/LLM framework for vector embedding, retrieval, and generation (we suggest Haystack, LlamaIndex, or LangChain).

Bonus points for:
- When creating a conversation, allow a selection of documents to be specified which will limit the context provided to the LLM for that conversation.
- Integration test(s) which check the correct functioning of the LLM chat endpoints.
- Dockerfile for running the API.
- Basic CI pipeline which checks code quality and runs tests.
- Python packaging and dependency management with poetry.

# Some tips!
We're not expecting a production grade solution to this problem! We want to see that you're a good coder and you're proficient in backend and ML work. Here are some tips to save some time. By all means feel free to ignore these suggestions; however, the task may take a lot longer! We know your time is precious and we respect that.
- If you are new to LLM frameworks, then we suggest using Haystack 2.0, as this is the simplest, with a more "on-rails" DX. Take a look at their [beginner RAG tutorial](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline), the [Weaviate document store docs](https://haystack.deepset.ai/integrations/weaviate-document-store#haystack-20), and the [PDF converter component docs](https://docs.haystack.deepset.ai/docs/pypdftodocument).
- Use the Weaviate and Ollama services in the `docker-compose.yml` to provide the vector db and LLM infrastructure which the app will require.
- Use an auto-generated Weaviate schema for the embeddings.
- If you change embeddings models or schema details, Weaviate may throw errors. Just delete the persistent volume for weaviate and restart in that case: `docker compose rm -f ; docker volume rm tunedd-backend-challenge_tunedd-weaviate-data`.
- Use Ollama with the suggested model for embeddings, `all-minilm:l6`, specified in the `.env` file. Many LLM frameworks support Ollama. Otherwise, use a service like Huggingface with `all-minilm` or `nomic-embed-text` etc or the OpenAI API.
- If you have a GPU you could use Ollama with the suggested model for chat, `phi3:3.8b-mini-instruct-4k-q4_K_M`, specified in the `.env`. See the `docker-compose.gpu.override.yml` file. Otherwise, use a service like Huggingface with `llama3-8b` or `mistral-7b` etc or the OpenAI API.
- The suggested embedding model gets pulled when the `ollama` service starts. Other models available in Ollama can be downloaded with the command: `docker exec tunedd-backend-challenge-ollama-1 ollama pull <model>`.
- Use a simple, non-durable, in-memory database, i.e., a global scoped Python dict or a dict wrapped by a class, for storing/retrieving the conversations and messages.
- Don't worry about authentication and authorisation.
- Something simple for loading in the data is fine, such as loading it in module scope when defining the vector db; no need for a separate out of band ETL process.
- There seems to be a dependency conflict with the latest version of `fastapi` and `llama-index-core`, so you might want to pin `fastapi==0.110.3` if you choose to use LlamaIndex.

# Assessment
We are looking for senior candidates with excellent knowledge of data structures, algorithms, and distributed systems, who take pride in producing code of a high standard, following software engineering best practices as well as specific Python conventions. The successful candidate should also be able to demonstrate a keen interest and competence in ML/AI generally (experience with LLMs specifically is a bonus). When assessing your submission we will be considering:
- Adherence to software engineering best practices and object oriented design principles such as SOLID, DRY etc.
- Idiomatic code which follows Python conventions and practices.
- High standard of code formatting and cleanliness as determined with automated tools (i.e. flake8 etc).
- Competence using popular ML/LLM frameworks.
- Following good practices in application testing (TDD, unittest/pytest, test fixtures, parameterised tests etc)
- Good commit hygiene and VCS practices (i.e. following Conventional Commits and using a popular branching model such as GitFlow, Scaled Trunk Based).
- Understanding and use of modern approaches to Python packaging (i.e. use of poetry and pyproject.toml).
- Understanding and use of best practices for building Docker containers, if applicable.
