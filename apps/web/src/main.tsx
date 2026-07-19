import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AtlasLensApp } from "./App";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("AtlasLens root element is missing");

createRoot(root).render(
  <StrictMode>
    <AtlasLensApp />
  </StrictMode>,
);
