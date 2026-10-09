'use client'

import { useRef, useState } from 'react'

interface FileUploadProps {
  onFileSelect: (file: File) => void
}

export default function FileUpload({ onFileSelect }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragActive, setIsDragActive] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true)
    } else if (e.type === 'dragleave') {
      setIsDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    const files = e.dataTransfer.files
    if (files && files[0]) {
      const file = files[0]
      if (file.name.endsWith('.edf')) {
        setFileName(file.name)
        onFileSelect(file)
      }
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files[0]) {
      const file = files[0]
      if (file.name.endsWith('.edf')) {
        setFileName(file.name)
        onFileSelect(file)
      }
    }
  }

  const handleClick = () => {
    inputRef.current?.click()
  }

  return (
    <div
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={handleClick}
      className={`
        relative border-2 border-dashed rounded-3xl p-12 text-center cursor-pointer
        transition-all duration-300 ease-out
        ${isDragActive 
          ? 'border-primary bg-primary/5 scale-105' 
          : 'border-border hover:border-primary/60 hover:bg-surface-secondary'
        }
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".edf"
        onChange={handleChange}
        className="hidden"
        aria-label="Upload EEG file"
      />

      {/* SVG illustration - brain waves */}
      <div className="mb-6 flex justify-center">
        <svg
          width="80"
          height="80"
          viewBox="0 0 80 80"
          fill="none"
          className="text-primary"
        >
          <path
            d="M 10 40 Q 20 30 30 40 T 50 40 T 70 40"
            stroke="currentColor"
            strokeWidth="2"
            fill="none"
            strokeLinecap="round"
          />
          <path
            d="M 10 50 Q 20 45 30 50 T 50 50 T 70 50"
            stroke="currentColor"
            strokeWidth="2"
            fill="none"
            opacity="0.6"
            strokeLinecap="round"
          />
          <path
            d="M 10 30 Q 20 25 30 30 T 50 30 T 70 30"
            stroke="currentColor"
            strokeWidth="2"
            fill="none"
            opacity="0.6"
            strokeLinecap="round"
          />
          {/* Small dots representing neural activity */}
          <circle cx="25" cy="40" r="2" fill="currentColor" />
          <circle cx="45" cy="40" r="2" fill="currentColor" />
          <circle cx="65" cy="40" r="2" fill="currentColor" />
        </svg>
      </div>

      <h3 className="font-serif text-xl mb-2 text-text">
        {fileName ? 'Ready to analyze' : 'Drop your EEG file here'}
      </h3>
      
      <p className="text-sm text-text-secondary mb-4">
        {fileName ? (
          <span className="font-medium">{fileName}</span>
        ) : (
          <>
            or <span className="text-primary font-medium">browse</span> to select
          </>
        )}
      </p>

      <p className="text-xs text-text-tertiary">
        Supported format: .edf (European Data Format)
      </p>

      {fileName && (
        <div className="mt-6 flex items-center justify-center gap-2 text-primary">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          <span className="text-sm font-medium">File ready for analysis</span>
        </div>
      )}
    </div>
  )
}
