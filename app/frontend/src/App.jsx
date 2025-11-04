import { useMemo, useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
} from "@mui/material";
import { Brightness4, Brightness7, Android } from "@mui/icons-material";
import Home from "./pages/Home";

function MainLayout({ children, toggleMode, mode }) {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh", 
        minWidth: "100vw", // fills full screen height
      }}
    >
      <AppBar position="static" color="primary" enableColorOnDark>
        <Toolbar>
          <Android sx={{ mr: 1 }} />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            IA vs. Humano
          </Typography>
          <IconButton color="inherit" onClick={toggleMode}>
            {mode === "dark" ? <Brightness7 /> : <Brightness4 />}
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Main content area fills remaining space */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          // alignItems: "center",
          // justifyContent: "center",
          px: 2,
          py: 4,
          width: "100%",
        }}
      >
        {children}
      </Box>
    </Box>
  );
}

function App() {
  const [mode, setMode] = useState("light");

  const toggleMode = () =>
    setMode((prevMode) => (prevMode === "light" ? "dark" : "light"));

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: { main: "#1976d2" },
          secondary: { main: "#9c27b0" },
        },
        typography: { fontFamily: "'Roboto', sans-serif" },
      }),
    [mode]
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <MainLayout toggleMode={toggleMode} mode={mode}>
          <Routes>
            <Route path="/" element={<Home />} />
          </Routes>
        </MainLayout>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
