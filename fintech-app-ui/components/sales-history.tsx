"use client"
import { useState, useEffect } from "react"
import { TrendingUp, Users, Receipt, DollarSign, ArrowUpRight, ArrowDownRight, Loader2 } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts"
import { api } from "@/lib/api"
import type { Transaction } from "@/lib/api"

const timePeriods = ["Today", "Week", "Month", "Year"]

export function SalesHistory() {
  const [activePeriod, setActivePeriod] = useState("Today")
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [usersData, setUsersData] = useState<any[]>([])
  const [totalUsers, setTotalUsers] = useState("?")
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchTransactions()
  }, [activePeriod])

  const fetchTransactions = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await api.getUserTransactions("all")
      setTransactions(response.transactions)
      
      fetchTotalUsers()
    } catch (err) {
      setError("Failed to fetch sales data")
      console.error("Failed to fetch transactions:", err)
    } finally {
      setIsLoading(false)
    }
  }

const fetchTotalUsers = async () => {
    try {
      const response = await fetch('http://localhost:8000/users')
      if (!response.ok) {
        console.error('Users API error:', response.status)
        setTotalUsers("?")
        return
      }

      const data = await response.json()
      if (data.success && data.users) {
        setUsersData(data.users)
        setTotalUsers(data.count || 0)
      } else {
        setTotalUsers("?")
      }
    } catch (err) {
      console.error("Failed to fetch user count:", err)
      setTotalUsers("?")
    }
  }

  const calculatePercentageChange = (current: number, previous: number): { change: string; trend: "up" | "down" } => {
    if (previous === 0) {
      return { change: current > 0 ? "+100%" : "0%", trend: current > 0 ? "up" : "up" }
    }
    const percentChange = ((current - previous) / previous) * 100
    const rounded = Math.round(percentChange * 10) / 10
    const sign = rounded >= 0 ? "+" : ""
    return {
      change: `${sign}${rounded}%`,
      trend: rounded >= 0 ? "up" : "down"
    }
  }

  const getFilteredTransactions = (period: string, subtractPeriod = false) => {
    const now = new Date()
    let startDate: Date
    let previousStartDate: Date | null = null

    switch (period) {
      case "Today":
        startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate())
        if (subtractPeriod) {
          previousStartDate = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1)
        }
        break
      case "Week":
        const dayOfWeek = now.getDay()
        const diff = now.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1)
        startDate = new Date(now.getFullYear(), now.getMonth(), diff)
        if (subtractPeriod) {
          previousStartDate = new Date(startDate)
          previousStartDate.setDate(previousStartDate.getDate() - 7)
        }
        break
      case "Month":
        startDate = new Date(now.getFullYear(), now.getMonth(), 1)
        if (subtractPeriod) {
          previousStartDate = new Date(now.getFullYear(), now.getMonth() - 1, 1)
        }
        break
      case "Year":
        startDate = new Date(now.getFullYear(), 0, 1)
        if (subtractPeriod) {
          previousStartDate = new Date(now.getFullYear() - 1, 0, 1)
        }
        break
      default:
        startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate())
    }

    const endDate = subtractPeriod && previousStartDate ? startDate : new Date(now)

    return transactions.filter(t => {
      if (!t.timestamp) return false
      const txnDate = new Date(t.timestamp)
      const start = subtractPeriod && previousStartDate ? previousStartDate : startDate
      return txnDate >= start && txnDate <= endDate
    })
  }

  const getFilteredUsers = (period: string, subtractPeriod = false) => {
    const now = new Date()
    let startDate: Date
    let previousStartDate: Date | null = null

    switch (period) {
      case "Today":
        startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate())
        if (subtractPeriod) {
          previousStartDate = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1)
        }
        break
      case "Week":
        const dayOfWeek = now.getDay()
        const diff = now.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1)
        startDate = new Date(now.getFullYear(), now.getMonth(), diff)
        if (subtractPeriod) {
          previousStartDate = new Date(startDate)
          previousStartDate.setDate(previousStartDate.getDate() - 7)
        }
        break
      case "Month":
        startDate = new Date(now.getFullYear(), now.getMonth(), 1)
        if (subtractPeriod) {
          previousStartDate = new Date(now.getFullYear(), now.getMonth() - 1, 1)
        }
        break
      case "Year":
        startDate = new Date(now.getFullYear(), 0, 1)
        if (subtractPeriod) {
          previousStartDate = new Date(now.getFullYear() - 1, 0, 1)
        }
        break
      default:
        startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate())
    }

    const endDate = subtractPeriod && previousStartDate ? startDate : new Date(now)

    return usersData.filter(u => {
      if (!u.created_at) return false
      const userDate = new Date(u.created_at)
      const start = subtractPeriod && previousStartDate ? previousStartDate : startDate
      return userDate >= start && userDate <= endDate
    })
  }

  const calculateStats = () => {
    if (transactions.length === 0) {
      return {
        totalRevenue: "KES 0.00",
        transactionCount: 0,
        revenueChange: { change: "0%", trend: "up" as const },
        transactionChange: { change: "0%", trend: "up" as const },
        userChange: { change: "0%", trend: "up" as const }
      }
    }

    const currentPeriodTransactions = getFilteredTransactions(activePeriod, false)
    const previousPeriodTransactions = getFilteredTransactions(activePeriod, true)

    const currentUsers = getFilteredUsers(activePeriod, false)
    const previousUsers = getFilteredUsers(activePeriod, true)

    const currentCompleted = currentPeriodTransactions.filter(t => t.status === "completed")
    const previousCompleted = previousPeriodTransactions.filter(t => t.status === "completed")

    const currentRevenue = currentCompleted.reduce((sum, t) => sum + t.amount, 0)
    const previousRevenue = previousCompleted.reduce((sum, t) => sum + t.amount, 0)

    const currentTransactions = currentCompleted.length
    const previousTransactions = previousCompleted.length

    const currentUserCount = currentUsers.length
    const previousUserCount = previousUsers.length

    return {
      totalRevenue: `KES ${currentRevenue.toLocaleString("en-KE", { minimumFractionDigits: 2 })}`,
      transactionCount: currentTransactions,
      revenueChange: calculatePercentageChange(currentRevenue, previousRevenue),
      transactionChange: calculatePercentageChange(currentTransactions, previousTransactions),
      userChange: calculatePercentageChange(currentUserCount, previousUserCount)
    }
  }

  const generateChartData = () => {
    const hours = ["9am", "10am", "11am", "12pm", "1pm", "2pm", "3pm", "4pm", "5pm", "6pm", "7pm"]

    const hourlyData = hours.map(hour => ({ time: hour, amount: 0 }))

    transactions
      .filter(t => t.status === "completed" && t.timestamp)
      .forEach(txn => {
        try {
          const timestamp = new Date(txn.timestamp!)
          const hour = timestamp.getHours()
          const hourLabel = hour < 12 ? `${hour}am` : hour === 12 ? "12pm" : `${hour - 12}pm`

          const dataPoint = hourlyData.find(d => d.time === hourLabel)
          if (dataPoint) {
            dataPoint.amount += txn.amount
          }
        } catch (e) {
          console.error("Error parsing transaction timestamp:", txn.timestamp)
        }
      })

    return hourlyData
  }

  const stats = calculateStats()
  const chartData = generateChartData()

  const statCards = [
    {
      title: "Total Revenue",
      value: stats.totalRevenue,
      change: stats.revenueChange.change,
      trend: stats.revenueChange.trend,
      icon: DollarSign,
    },
    {
      title: "Transactions",
      value: stats.transactionCount.toString(),
      change: stats.transactionChange.change,
      trend: stats.transactionChange.trend,
      icon: Receipt,
    },
    {
      title: "Registered Users",
      value: totalUsers.toString(),
      change: stats.userChange.change,
      trend: stats.userChange.trend,
      icon: Users,
    },
  ]

  if (isLoading) {
    return (
      <div className="p-4 lg:p-6 space-y-6">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-[#ccff00]" />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4 lg:p-6 space-y-6">
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-8 text-center">
          <p className="text-red-400">{error}</p>
          <button onClick={fetchTransactions} className="mt-4 text-[#ccff00] hover:underline">
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 lg:p-6 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h1 className="text-2xl font-semibold text-white">Sales Analytics</h1>

        <div className="flex bg-[#18181b] rounded-lg p-1 border border-[#27272a]">
          {timePeriods.map((period) => (
            <button
              key={period}
              onClick={() => setActivePeriod(period)}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
                activePeriod === period ? "bg-[#ccff00] text-black" : "text-[#a1a1aa] hover:text-white"
              }`}
            >
              {period}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {statCards.map((stat) => (
          <Card key={stat.title} className="bg-[#18181b] border-[#27272a]">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="w-10 h-10 rounded-lg bg-[#27272a] flex items-center justify-center">
                  <stat.icon className="h-5 w-5 text-[#a1a1aa]" />
                </div>
                <div className={`flex items-center gap-1 text-xs font-medium ${stat.trend === "up" ? "text-[#ccff00]" : "text-red-500"}`}>
                  {stat.trend === "up" ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
                  {stat.change}
                </div>
              </div>
              <p className="text-xs text-[#a1a1aa] mb-1">{stat.title}</p>
              <p className="text-xl font-semibold font-mono text-[#ccff00]">{stat.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="bg-[#18181b] border-[#27272a]">
        <CardHeader className="pb-2">
          <CardTitle className="text-white text-lg">Revenue Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="neonGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ccff00" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#ccff00" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis dataKey="time" stroke="#71717a" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis
                  stroke="#71717a"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => `${value / 1000}k`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#18181b",
                    border: "1px solid #27272a",
                    borderRadius: "8px",
                    color: "#fff",
                  }}
                  labelStyle={{ color: "#a1a1aa" }}
                  formatter={(value: number) => [`KES ${value.toLocaleString()}`, "Revenue"]}
                />
                <Area type="monotone" dataKey="amount" stroke="#ccff00" strokeWidth={2} fill="url(#neonGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-[#18181b] border-[#27272a]">
        <CardHeader className="pb-2">
          <CardTitle className="text-white text-lg">Recent Transactions</CardTitle>
        </CardHeader>
        <CardContent>
          {transactions.length === 0 ? (
            <div className="text-center py-8 text-[#52525b]">No transactions found</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[#27272a]">
                    <th className="text-left text-xs font-medium text-[#a1a1aa] pb-3">Transaction ID</th>
                    <th className="text-left text-xs font-medium text-[#a1a1aa] pb-3">User ID</th>
                    <th className="text-left text-xs font-medium text-[#a1a1aa] pb-3">Phone</th>
                    <th className="text-right text-xs font-medium text-[#a1a1aa] pb-3">Amount</th>
                    <th className="text-right text-xs font-medium text-[#a1a1aa] pb-3">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {transactions.slice(0, 10).map((txn) => (
                    <tr key={txn.checkout_request_id || txn.merchant_request_id} className="border-b border-[#27272a] last:border-0">
                      <td className="py-3 text-sm font-mono text-[#a1a1aa]">
                        {txn.checkout_request_id || txn.merchant_request_id || "N/A"}
                      </td>
                      <td className="py-3 text-sm text-white">{txn.user_id}</td>
                      <td className="py-3 text-sm text-white font-mono">{txn.phone_number}</td>
                      <td className="py-3 text-sm font-mono text-[#ccff00] text-right">
                        KES {txn.amount.toLocaleString()}.00
                      </td>
                      <td className="py-3 text-right">
                        <Badge
                          className={`${
                            txn.status === "completed"
                              ? "bg-[#ccff00] text-black hover:bg-[#ccff00]/90"
                              : txn.status === "pending"
                                ? "bg-yellow-500/20 text-yellow-500 border border-yellow-500/50"
                                : "bg-red-500/20 text-red-500 border border-red-500/50"
                          } font-medium`}
                        >
                          {txn.status}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default SalesHistory