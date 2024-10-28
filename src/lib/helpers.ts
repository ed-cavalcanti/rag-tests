import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import type { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import {
  RunnablePassthrough,
  RunnableSequence,
  RunnableWithMessageHistory,
} from "@langchain/core/runnables";
import type { VectorStoreRetriever } from "@langchain/core/vectorstores";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { HttpResponseOutputParser } from "langchain/output_parsers";

type loadAndSplitChunkProps = {
  chunkSize: number;
  chunkOverlap: number;
  docPath: string;
};

export const loadAndSplitChunks = async ({
  chunkSize,
  chunkOverlap,
  docPath,
}: loadAndSplitChunkProps): Promise<Document<Record<string, any>>[]> => {
  const loader = new PDFLoader(docPath);
  const rawDoc = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkOverlap,
    chunkSize,
  });

  const splitDocs = await splitter.splitDocuments(rawDoc);
  return splitDocs;
};

export const initializeVectorStore = async (
  documents: Document<Record<string, any>>[]
): Promise<MemoryVectorStore> => {
  const embeddingModel = new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004",
    taskType: TaskType.RETRIEVAL_DOCUMENT,
  });
  const vectorStore = new MemoryVectorStore(embeddingModel);
  await vectorStore.addDocuments(documents);
  return vectorStore;
};

const convertDocsToString = (documents: Document[]): string => {
  return documents
    .map((document) => {
      return `<doc>\n${document.pageContent}\n</doc>`;
    })
    .join("\n");
};

export const createDocumentRetrievalChain = (
  retriever: VectorStoreRetriever<MemoryVectorStore>
): RunnableSequence<any, string> => {
  return RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString,
  ]);
};

export const createRetrievalQuestionChain = (): RunnableSequence<
  any,
  string
> => {
  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Dada a seguinte conversa e uma pergunta de acompanhamento,
  reformule a pergunta de acompanhamento para ser uma pergunta independente.`;

  const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
      "human",
      "Reformule a seguinte pergunta como uma pergunta independente:\n{question}",
    ],
  ]);

  return RunnableSequence.from([
    rephraseQuestionChainPrompt,
    new ChatGoogleGenerativeAI({ model: "gemini-1.5-pro", temperature: 0.1 }),
    new StringOutputParser(),
  ]);
};

interface MessageHistories {
  [key: string]: ChatMessageHistory;
}

const createFinalRetrievalChain = async () => {
  const splitDocs = await loadAndSplitChunks({
    chunkSize: 1536,
    chunkOverlap: 128,
    docPath:
      "C:\\Users\\Admin\\Documents\\Estágio\\rag\\src\\data\\docrag01.pdf",
  });

  const vectorStore = await initializeVectorStore(splitDocs);
  const retriever = vectorStore.asRetriever();

  const documentRetrievalChain = createDocumentRetrievalChain(retriever);
  const rephraseQuestionChain = createRetrievalQuestionChain();

  const ANSWER_CHAIN_SYSTEM_TEMPLATE = `Você é um pesquisador experiente,
  especialista em interpretar e responder perguntas com base nas fontes fornecidas.
  Usando o contexto e histórico de bate-papo fornecidos abaixo, 
  responda à pergunta do usuário da melhor maneira possível
  usando apenas os recursos fornecidos. Seja prolixo!

  <context>
    {context}
  </context>`;

  const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
      "human",
      `Agora, responda a esta pergunta usando o contexto anterior e o histórico de bate-papo:
    
      {standalone_question}`,
    ],
  ]);

  const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseQuestionChain,
    }),
    RunnablePassthrough.assign({
      context: documentRetrievalChain,
    }),
    answerGenerationChainPrompt,
    new ChatGoogleGenerativeAI({ model: "gemini-1.5-pro" }),
  ]);

  const httpResponseOutputParser = new HttpResponseOutputParser({
    contentType: "text/plain",
  });

  const messageHistories: MessageHistories = {};
  const getMessageHistoryForSession = (sessionId: string) => {
    if (messageHistories[sessionId] !== undefined) {
      return messageHistories[sessionId];
    }
    const newChatSessionHistory = new ChatMessageHistory();
    messageHistories[sessionId] = newChatSessionHistory;
    return newChatSessionHistory;
  };

  const finalRetrievalChain = new RunnableWithMessageHistory({
    runnable: conversationalRetrievalChain,
    getMessageHistory: getMessageHistoryForSession,
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  }).pipe(httpResponseOutputParser);

  return finalRetrievalChain;
};

type handlerProps = {
  question: string;
  sessionId: string;
};

export const handler = async ({ question, sessionId }: handlerProps) => {
  const finalRetrievalChain = await createFinalRetrievalChain();
  const stream = await finalRetrievalChain.stream(
    {
      question: question,
    },
    { configurable: { sessionId } }
  );
  return stream;
};
