import { useState, useEffect, useCallback } from "react";
import { Box, Typography, Tabs, Tab, CircularProgress } from "@mui/material";
import TabTextos from "./TabTextos";
import TabJogo from "./TabJogo";

function Home() {
  const [tabIndex, setTabIndex] = useState(0);
  const [texts, setTexts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchTexts = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("http://localhost:8080/dataset/random25");

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data && Array.isArray(data.samples)) {
        setTexts(data.samples);
      } else {
        console.error("Unexpected response format:", data);
        setTexts([]);
      }
    } catch (err) {
      console.error("Error fetching texts:", err);
      setError("Erro ao carregar o dataset. Verifique o servidor.");
      setTexts([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTexts();
  }, [fetchTexts]);

  const handleTabChange = (event, newValue) => {
    setTabIndex(newValue);
  };

  return (
    <Box
      sx={{
        textAlign: "center",
        mt: 6,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <Typography variant="h4" gutterBottom>
        Reconhecedor de Textos
      </Typography>

      <Tabs value={tabIndex} onChange={handleTabChange} centered sx={{ mb: 3 }}>
        <Tab label="Textos" />
        <Tab label="Jogo" />
      </Tabs>

      {loading ? (
        <Box sx={{ mt: 4 }}>
          <CircularProgress />
          <Typography variant="body1" sx={{ mt: 2 }}>
            Carregando textos...
          </Typography>
        </Box>
      ) : error ? (
        <Typography color="error" sx={{ mt: 4 }}>
          {error}
        </Typography>
      ) : (
        <>
          {tabIndex === 0 && <TabTextos />}
          {tabIndex === 1 && <TabJogo texts={texts} onReplay={fetchTexts} />}
        </>
      )}
    </Box>
  );
}

export default Home;
