import { BrowserRouter, Route, Routes } from "react-router-dom";
import { WarehouseProvider } from "./context/WarehouseContext";
import Layout from "./components/Layout";
import OperationsPage from "./pages/OperationsPage";
import TrustPage from "./pages/TrustPage";
import MemoryPage from "./pages/MemoryPage";
import ArchitecturePage from "./pages/ArchitecturePage";

export default function App() {
  return (
    <WarehouseProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<OperationsPage />} />
            <Route path="trust" element={<TrustPage />} />
            <Route path="memory" element={<MemoryPage />} />
            <Route path="architecture" element={<ArchitecturePage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </WarehouseProvider>
  );
}
