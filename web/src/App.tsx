import { BrowserRouter, Route, Routes } from "react-router-dom";
import { WarehouseProvider } from "./context/WarehouseContext";
import Layout from "./components/Layout";
import DashboardPage from "./pages/DashboardPage";
import OperationsPage from "./pages/OperationsPage";
import EpistemicTestsPage from "./pages/EpistemicTestsPage";
import TrustPage from "./pages/TrustPage";
import MemoryPage from "./pages/MemoryPage";
import ArchitecturePage from "./pages/ArchitecturePage";

export default function App() {
  return (
    <WarehouseProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<DashboardPage />} />
            <Route path="operations" element={<OperationsPage />} />
            <Route path="epistemic" element={<EpistemicTestsPage />} />
            <Route path="trust" element={<TrustPage />} />
            <Route path="memory" element={<MemoryPage />} />
            <Route path="architecture" element={<ArchitecturePage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </WarehouseProvider>
  );
}
