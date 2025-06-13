"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Upload, ImageIcon, Info, History, Trash2, ChevronRight, AlertCircle, CheckCircle2 } from "lucide-react"
import Image from "next/image"
import { cn } from "@/lib/utils"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

// Types
interface ClassificationResult {
  id: string
  class: string
  confidence: number
  timestamp: string
  imageUrl: string
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<ClassificationResult[]>([])
  const [activeTab, setActiveTab] = useState("upload")
  const [dragActive, setDragActive] = useState(false)

  // Load history from localStorage on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem("classificationHistory")
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory))
      } catch (e) {
        console.error("Failed to parse history from localStorage")
      }
    }
  }, [])

  // Save history to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem("classificationHistory", JSON.stringify(history))
  }, [history])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      processFile(selectedFile)
    }
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const processFile = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      setError("Please upload an image file")
      return
    }

    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setError(null)
  }

  const handleSubmit = async () => {
    if (!file) return

    setIsLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      // In a real application, this would be your API endpoint
      const response = await fetch("http://localhost:8000/classify", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Classification failed")
      }

      const data = await response.json()

      // Create a new result with additional metadata
      const newResult: ClassificationResult = {
        id: Date.now().toString(),
        class: data.class,
        confidence: data.confidence,
        timestamp: new Date().toISOString(),
        imageUrl: preview || "",
      }

      setResult(newResult)

      // Add to history
      setHistory((prev) => [newResult, ...prev].slice(0, 10)) // Keep only the 10 most recent
    } catch (err) {
      setError("An error occurred during classification. Please try again.")
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  const clearHistory = () => {
    setHistory([])
  }

  const removeHistoryItem = (id: string) => {
    setHistory((prev) => prev.filter((item) => item.id !== id))
  }

  const viewHistoryItem = (item: ClassificationResult) => {
    setPreview(item.imageUrl)
    setResult(item)
    setActiveTab("upload")
  }

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <ImageIcon className="h-8 w-8 text-emerald-600 dark:text-emerald-400" />
              <h1 className="text-2xl font-bold">Landscape Painting Classifier</h1>
            </div>
            <Badge variant="outline" className="px-3 py-1">
              AI-Powered
            </Badge>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Panel */}
          <div className="lg:col-span-2">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="upload" className="flex items-center gap-2">
                  <Upload className="h-4 w-4" />
                  <span>Upload & Classify</span>
                </TabsTrigger>
                <TabsTrigger value="history" className="flex items-center gap-2">
                  <History className="h-4 w-4" />
                  <span>History</span>
                  {history.length > 0 && (
                    <Badge variant="secondary" className="ml-1">
                      {history.length}
                    </Badge>
                  )}
                </TabsTrigger>
              </TabsList>

              <TabsContent value="upload" className="mt-6 space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Upload a Landscape Painting</CardTitle>
                    <CardDescription>
                      Upload a Chinese or Japanese landscape painting for classification
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div
                      className={cn(
                        "flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-12 text-center transition-colors",
                        dragActive
                          ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20"
                          : "border-gray-300 dark:border-gray-700",
                      )}
                      onDragEnter={handleDrag}
                      onDragLeave={handleDrag}
                      onDragOver={handleDrag}
                      onDrop={handleDrop}
                    >
                      <input
                        type="file"
                        id="file-upload"
                        className="hidden"
                        accept="image/*"
                        onChange={handleFileChange}
                      />
                      <label htmlFor="file-upload" className="flex flex-col items-center justify-center cursor-pointer">
                        <ImageIcon className="h-12 w-12 text-gray-400 mb-4" />
                        <span className="text-lg font-medium mb-1">
                          {dragActive ? "Drop your image here" : "Click or drag to upload"}
                        </span>
                        <span className="text-sm text-gray-500">JPG, PNG, GIF up to 10MB</span>
                      </label>
                    </div>

                    {error && (
                      <Alert variant="destructive" className="mt-4">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>

                {preview && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Image Preview</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex flex-col items-center">
                        <div className="relative w-full h-64 overflow-hidden rounded-lg">
                          <Image src={preview || "/placeholder.svg"} alt="Preview" fill className="object-contain" />
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter className="flex justify-center">
                      {!result && (
                        <Button onClick={handleSubmit} disabled={isLoading} className="w-full sm:w-auto">
                          {isLoading ? "Analyzing..." : "Classify Image"}
                        </Button>
                      )}
                    </CardFooter>
                  </Card>
                )}

                {isLoading && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Analyzing Image</CardTitle>
                      <CardDescription>Our AI is analyzing the artistic style...</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Progress value={45} className="h-2" />
                    </CardContent>
                  </Card>
                )}

                {result && (
                  <Card
                    className={cn("border-l-4", result.class === "Chinese" ? "border-l-red-500" : "border-l-blue-500")}
                  >
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center gap-2">
                          <CheckCircle2 className="h-5 w-5 text-green-500" />
                          Classification Result
                        </CardTitle>
                        <Badge
                          className={cn(
                            result.class === "Chinese" ? "bg-red-100 text-red-800" : "bg-blue-100 text-blue-800",
                          )}
                        >
                          {result.class}
                        </Badge>
                      </div>
                      <CardDescription>{new Date(result.timestamp).toLocaleString()}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium">Confidence</span>
                            <span className="text-sm font-medium">{(result.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <Progress
                            value={result.confidence * 100}
                            className={cn("h-2", result.class === "Chinese" ? "bg-red-100" : "bg-blue-100")}
                          />
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setFile(null)
                          setPreview(null)
                          setResult(null)
                        }}
                      >
                        Classify Another Image
                      </Button>
                    </CardFooter>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="history" className="mt-6">
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between">
                    <div>
                      <CardTitle>Classification History</CardTitle>
                      <CardDescription>Your recent classification results</CardDescription>
                    </div>
                    {history.length > 0 && (
                      <Button variant="outline" size="sm" onClick={clearHistory} className="flex items-center gap-1">
                        <Trash2 className="h-4 w-4" />
                        <span>Clear</span>
                      </Button>
                    )}
                  </CardHeader>
                  <CardContent>
                    {history.length === 0 ? (
                      <div className="text-center py-8 text-gray-500">
                        <History className="h-12 w-12 mx-auto mb-3 opacity-30" />
                        <p>No classification history yet</p>
                        <p className="text-sm">Classified images will appear here</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {history.map((item) => (
                          <div
                            key={item.id}
                            className="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                          >
                            <div className="relative h-16 w-16 rounded overflow-hidden flex-shrink-0">
                              <Image
                                src={item.imageUrl || "/placeholder.svg"}
                                alt={item.class}
                                fill
                                className="object-cover"
                              />
                            </div>
                            <div className="flex-grow">
                              <div className="flex items-center gap-2">
                                <Badge
                                  className={cn(
                                    item.class === "Chinese" ? "bg-red-100 text-red-800" : "bg-blue-100 text-blue-800",
                                  )}
                                >
                                  {item.class}
                                </Badge>
                                <span className="text-sm text-gray-500">
                                  {(item.confidence * 100).toFixed(1)}% confidence
                                </span>
                              </div>
                              <p className="text-xs text-gray-500 mt-1">{new Date(item.timestamp).toLocaleString()}</p>
                            </div>
                            <div className="flex items-center gap-2">
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Button variant="ghost" size="icon" onClick={() => viewHistoryItem(item)}>
                                      <ChevronRight className="h-4 w-4" />
                                    </Button>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p>View details</p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>

                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Button variant="ghost" size="icon" onClick={() => removeHistoryItem(item.id)}>
                                      <Trash2 className="h-4 w-4" />
                                    </Button>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p>Remove from history</p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Side Panel - Only About This Classifier */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="h-5 w-5" />
                  About This Classifier
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  This application uses a ResNet18 deep learning model trained to distinguish between Chinese and
                  Japanese landscape paintings (산수화/山水画).
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  The model was trained on a dataset of landscape paintings and achieves approximately 92% accuracy in
                  distinguishing between the two styles.
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Upload an image of a landscape painting to see whether it's classified as Chinese or Japanese style.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              © {new Date().getFullYear()} Landscape Painting Classifier
            </p>
          </div>
        </div>
      </footer>
    </main>
  )
}
