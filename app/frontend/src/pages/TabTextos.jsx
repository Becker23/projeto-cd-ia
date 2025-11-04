import { useState } from "react";
import {
  Box,
  TextField,
  Button,
  CircularProgress,
  Typography,
  Paper,
  Divider,
} from "@mui/material";

function TabTextos() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://localhost:8080/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Erro ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Erro ao conectar à API:", err);
      setError("Não foi possível conectar ao servidor.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: "100%",
        maxWidth: 700,
        mx: "auto",
        mt: 4,
      }}
    >
      <TextField
        label="Digite ou cole seu texto aqui"
        multiline
        rows={6}
        fullWidth
        variant="outlined"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <Button
        variant="contained"
        onClick={handleSubmit}
        disabled={loading || !text.trim()}
        sx={{ mt: 2, width: 200, height: 45 }}
      >
        {loading ? <CircularProgress size={24} color="inherit" /> : "Analisar Texto"}
      </Button>

      {error && (
        <Typography color="error" sx={{ mt: 3 }}>
          {error}
        </Typography>
      )}

      {/* Results Section */}
      {result && (
        <Paper
          elevation={3}
          sx={{
            mt: 4,
            p: 3,
            width: "100%",
            maxWidth: 600,
            backgroundColor: "background.paper",
          }}
        >
          <Typography variant="h6" gutterBottom>
            Resultados:
          </Typography>

          {/* BERT result */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1">
              Modelo: <strong>{result.bert.model}</strong>
            </Typography>
            <Typography>
              Predição:{" "}
              <strong
                style={{
                  color:
                    result.bert.prediction.toLowerCase() === "ia"
                      ? "#1976d2"
                      : "#2e7d32",
                }}
              >
                {result.bert.prediction.toUpperCase()}
              </strong>
            </Typography>
            <Typography>Confiança: {result.bert.confidence.toFixed(2)}%</Typography>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* TF-IDF result */}
          <Box>
            <Typography variant="subtitle1">
              Modelo: <strong>{result.tfidf.model}</strong>
            </Typography>
            <Typography>
              Predição:{" "}
              <strong
                style={{
                  color:
                    result.tfidf.prediction.toLowerCase() === "ia"
                      ? "#1976d2"
                      : "#2e7d32",
                }}
              >
                {result.tfidf.prediction.toUpperCase()}
              </strong>
            </Typography>
            <Typography>Confiança: {result.tfidf.confidence.toFixed(2)}%</Typography>
          </Box>
        </Paper>
      )}
    </Box>
  );
}

export default TabTextos;
