import express from 'express';
import fs from 'fs';
import path from 'path';
import bodyParser from 'body-parser';
import cors from 'cors';

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(cors({
  origin: 'http://localhost:5173',
  methods: ['POST'],
}));

app.post('/save-results', (req, res) => {
  const { scores, traitScores } = req.body;
  const resultData = {
    scores,
    traitScores,
    timestamp: new Date().toISOString(),
  };

  const filePath = path.join(path.resolve(), 'test_result.json');
  fs.writeFileSync(filePath, JSON.stringify(resultData, null, 2));
  res.status(200).send('Results saved successfully');
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
