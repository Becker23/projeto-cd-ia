import { useState, useEffect } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
} from "@mui/material";

function TabJogo({ texts, onReplay }) {
  const [index, setIndex] = useState(0);
  const [feedbackColor, setFeedbackColor] = useState("");
  const [locked, setLocked] = useState(false);
  const [score, setScore] = useState(0);
  const [finished, setFinished] = useState(false);

  // Reset when new texts arrive
  useEffect(() => {
    setIndex(0);
    setScore(0);
    setFinished(false);
  }, [texts]);

  if (!texts || texts.length === 0) {
    return (
      <Box sx={{ textAlign: "center", mt: 4 }}>
        <CircularProgress />
        <Typography variant="body1" sx={{ mt: 2 }}>
          Carregando textos...
        </Typography>
      </Box>
    );
  }

  const handleChoice = (choice) => {
    if (locked || finished) return;
    setLocked(true);

    const currentText = texts[index];
    const correct = choice.toLowerCase() === currentText.classe.toLowerCase();
    if (correct) setScore((prev) => prev + 1);
    setFeedbackColor(correct ? "#4CAF50" : "#F44336");

    setTimeout(() => {
      setFeedbackColor("");
      setLocked(false);

      if (index + 1 >= texts.length) {
        setFinished(true);
      } else {
        setIndex((prev) => prev + 1);
      }
    }, 1000);
  };

  const handleReplay = async () => {
    setIndex(0);
    setScore(0);
    setFinished(false);
    setFeedbackColor("");
    await onReplay(); // 🔁 Fetch new texts from parent
  };

  if (finished) {
    return (
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: 300,
          width: 650,
          mt: 4,
          textAlign: "center",
        }}
      >
        <Typography variant="h5" gutterBottom>
          Obrigado por jogar!
        </Typography>
        <Typography variant="body1" sx={{ mb: 3 }}>
          Você acertou <strong>{score}</strong> de {texts.length} textos.
        </Typography>

        <Button variant="contained" color="primary" onClick={handleReplay}>
          Jogar novamente
        </Button>
      </Box>
    );
  }

  const currentText = texts[index];

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        mt: 4,
      }}
    >
      <Card
        sx={{
          width: 650,
          height: 300,
          transition: "background-color 0.3s ease",
          backgroundColor: feedbackColor || "background.paper",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
        }}
      >
        <CardContent
          sx={{
            display: "flex",
            flexDirection: "column",
            height: "100%",
            justifyContent: "space-between",
            p: 3,
          }}
        >
          {/* Scrollable text */}
          <Box
            sx={{
              flexGrow: 1,
              overflowY: "auto",
              pr: 1,
              mb: 2,
              "&::-webkit-scrollbar": {
                width: "6px",
              },
              "&::-webkit-scrollbar-thumb": {
                backgroundColor: "#bbb",
                borderRadius: "3px",
              },
            }}
          >
            <Typography variant="body1" sx={{ textAlign: "justify" }}>
              {currentText?.snippet_text || ""}
            </Typography>
          </Box>

          {/* Choice buttons */}
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              gap: 5,
            }}
          >
            <Button
              variant="contained"
              color="primary"
              onClick={() => handleChoice("ia")}
              disabled={locked}
              sx={{ width: 100, height: 40 }}
            >
              IA
            </Button>

            <Button
              variant="contained"
              color="secondary"
              onClick={() => handleChoice("humano")}
              disabled={locked}
              sx={{ width: 100, height: 40 }}
            >
              HUMANO
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Typography variant="body2" sx={{ mt: 2 }}>
        Texto {index + 1} de {texts.length}
      </Typography>
    </Box>
  );
}

export default TabJogo;
