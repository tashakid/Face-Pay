import { AppShell } from "@/components/app-shell"
import { SalesHistory } from "@/components/sales-history"

export default function SalesPage() {
  return (
    <AppShell activeTab="sales">
      <SalesHistory />
    </AppShell>
  )
}
