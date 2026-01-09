import { AppShell } from "@/components/app-shell"
import { EnrollCustomer } from "@/components/enroll-customer"

export default function EnrollPage() {
  return (
    <AppShell activeTab="enroll">
      <EnrollCustomer />
    </AppShell>
  )
}
