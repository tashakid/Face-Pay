"use client"

import type React from "react"

import { useState } from "react"
import { Home, UserPlus, History, ScanFace, Menu, X, Building2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface AppShellProps {
  children: React.ReactNode
  activeTab?: string
}

const navItems = [
  { id: "home", label: "Home", icon: Home, href: "/" },
  { id: "enroll", label: "Enroll Customer", icon: UserPlus, href: "/enroll" },
  { id: "sales", label: "Sales History", icon: History, href: "/sales" },
]

export function AppShell({ children, activeTab = "home" }: AppShellProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="min-h-screen bg-black flex flex-col">
      {/* Header - Dark theme */}
      <header className="bg-[#09090b] border-b border-[#27272a] sticky top-0 z-50">
        <div className="flex items-center justify-between px-4 h-16">
          {/* Mobile menu button */}
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden text-white hover:bg-[#18181b]"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>

          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-xl bg-[#ccff00] flex items-center justify-center">
              <ScanFace className="h-5 w-5 text-black" />
            </div>
            <span className="text-xl font-semibold text-white">FacePay</span>
          </div>

          <div className="flex items-center gap-4">
            {/* System Status - Neon lime */}
            <div className="hidden sm:flex items-center gap-2 text-sm text-[#a1a1aa]">
              <div className="w-2 h-2 rounded-full bg-[#ccff00] animate-pulse" />
              <span>Online</span>
            </div>

            {/* Business Name & Terminal ID */}
            <div className="text-right">
              <div className="flex items-center gap-2">
                <Building2 className="h-4 w-4 text-[#a1a1aa]" />
                <span className="text-sm font-medium text-white">Java House - Kimathi St</span>
              </div>
              <span className="text-xs font-mono text-[#ccff00]">Terminal ID: #T-804</span>
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-1">
        {/* Desktop Sidebar - Dark theme */}
        <aside className="hidden lg:flex w-64 bg-[#09090b] border-r border-[#27272a] flex-col fixed left-0 top-16 bottom-0">
          <nav className="flex-1 p-4 space-y-1">
            {navItems.map((item) => (
              <a
                key={item.id}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all",
                  activeTab === item.id
                    ? "bg-[#ccff00] text-black"
                    : "text-[#a1a1aa] hover:bg-[#18181b] hover:text-white",
                )}
              >
                <item.icon className="h-5 w-5" />
                {item.label}
              </a>
            ))}
          </nav>

          {/* Sidebar footer - Dark theme */}
          <div className="p-4 border-t border-[#27272a]">
            <div className="flex items-center gap-3 px-4 py-3 bg-[#18181b] rounded-xl border border-[#27272a]">
              <div className="w-10 h-10 rounded-full bg-[#ccff00]/10 flex items-center justify-center">
                <ScanFace className="h-5 w-5 text-[#ccff00]" />
              </div>
              <div>
                <p className="text-sm font-medium text-white">POS Active</p>
                <p className="text-xs text-[#a1a1aa]">Ready for payments</p>
              </div>
            </div>
          </div>
        </aside>

        {/* Mobile Sidebar Overlay */}
        {sidebarOpen && (
          <div className="lg:hidden fixed inset-0 bg-black/80 z-40" onClick={() => setSidebarOpen(false)} />
        )}

        {/* Mobile Sidebar - Dark theme */}
        <aside
          className={cn(
            "lg:hidden fixed left-0 top-16 bottom-0 w-72 bg-[#09090b] border-r border-[#27272a] z-50 transform transition-transform duration-300",
            sidebarOpen ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <nav className="flex-1 p-4 space-y-1">
            {navItems.map((item) => (
              <a
                key={item.id}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all",
                  activeTab === item.id
                    ? "bg-[#ccff00] text-black"
                    : "text-[#a1a1aa] hover:bg-[#18181b] hover:text-white",
                )}
                onClick={() => setSidebarOpen(false)}
              >
                <item.icon className="h-5 w-5" />
                {item.label}
              </a>
            ))}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 lg:ml-64 pb-20 lg:pb-0">{children}</main>
      </div>

      {/* Mobile Bottom Tab Bar - Dark theme */}
      <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-[#09090b] border-t border-[#27272a] z-50">
        <div className="flex items-center justify-around h-16">
          {navItems.map((item) => (
            <a
              key={item.id}
              href={item.href}
              className={cn(
                "flex flex-col items-center gap-1 px-4 py-2 rounded-lg transition-all",
                activeTab === item.id ? "text-[#ccff00]" : "text-[#a1a1aa]",
              )}
            >
              <item.icon className={cn("h-5 w-5", activeTab === item.id && "stroke-[2.5]")} />
              <span className="text-xs font-medium">{item.label}</span>
            </a>
          ))}
        </div>
      </nav>
    </div>
  )
}
