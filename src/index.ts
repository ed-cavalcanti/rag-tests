import "dotenv/config";
import express, { Request, Response } from "express";
import { handler } from "./lib/helpers";

const app = express();
const port = process.env.API_PORT || 8087

app.use(express.json());

app.post("/generate", async (req: Request, res: Response) => {
  const { question, session_id } = req.body;

  if (!question || !session_id) {
    res.status(400).json({ error: "question and session_id are required" });
  }

  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Transfer-Encoding", "chunked");

  const aiResponseStream = await handler({ question, sessionId: session_id });

  for await (const chunk of aiResponseStream) {
    const textChunk = new TextDecoder().decode(chunk);
    res.write(textChunk);
  }

  res.end();
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
