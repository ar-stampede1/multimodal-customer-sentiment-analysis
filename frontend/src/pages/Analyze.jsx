import { useState } from 'react'
import {
  Upload,
  Image as ImageIcon,
  BrainCircuit,
  X,
  Loader2,
  Sparkles,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react'

import { predictSentiment } from '../services/api'

// High-contrast, bright, eye-catching badge styles without blues
const sentimentStyles = {
  'Very Dissatisfied': 'text-rose-700 bg-rose-50 border-rose-200',
  'Dissatisfied': 'text-orange-700 bg-orange-50 border-orange-200',
  'Neutral': 'text-amber-700 bg-amber-50 border-amber-200',
  'Satisfied': 'text-emerald-700 bg-emerald-50 border-emerald-200',
  'Very Satisfied': 'text-violet-700 bg-violet-50 border-violet-200',
}

const classNames = [
  'Very Dissatisfied',
  'Dissatisfied',
  'Neutral',
  'Satisfied',
  'Very Satisfied',
]

function Analyze() {
  const [reviewTitle, setReviewTitle] = useState('')
  const [reviewText, setReviewText] = useState('')
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleImageChange = (event) => {
    const file = event.target.files?.[0]

    if (!file) return

    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setError('Only JPEG, PNG and WebP images are supported.')
      event.target.value = ''
      return
    }

    if (file.size > 5 * 1024 * 1024) {
      setError('Image size must not exceed 5 MB.')
      event.target.value = ''
      return
    }

    setError('')
    setImage(file)
    setImagePreview(URL.createObjectURL(file))
    setResult(null)
  }

  const removeImage = () => {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview)
    }

    setImage(null)
    setImagePreview('')
  }

  const handleSubmit = async (event) => {
    event.preventDefault()

    if (!reviewTitle.trim() && !reviewText.trim() && !image) {
      setError('Please enter review text or upload an image.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const data = await predictSentiment({
        reviewTitle,
        reviewText,
        image,
      })

      setResult(data)
    } catch (err) {
      console.error('Prediction error:', err)

      setError(
        err.response?.data?.detail ||
        'Unable to analyze the review. Please check your backend connection.'
      )
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setReviewTitle('')
    setReviewText('')
    removeImage()
    setResult(null)
    setError('')
  }

  return (
    <div className="space-y-8">
      {/* Page Heading */}
      <div>
        <div className="flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-100 text-violet-700 shadow-xs">
            <BrainCircuit size={22} />
          </div>

          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Analyze Sentiment
          </h1>
        </div>

        <p className="mt-1 text-sm text-slate-500">
          Run inference across text, product images, or fused multimodal inputs with instant probability breakdown.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        {/* Input Form */}
        <form
          onSubmit={handleSubmit}
          className="space-y-6 rounded-2xl border border-purple-100/90 bg-white p-6 sm:p-7 shadow-xs"
        >
          <div>
            <h2 className="text-lg font-bold text-slate-900 tracking-tight">
              Review Input
            </h2>
            <p className="mt-0.5 text-xs text-slate-500 font-medium">
              Provide textual review details and optional product imagery
            </p>
          </div>

          {/* Review Title */}
          <div>
            <label className="mb-2 block text-xs font-bold uppercase tracking-wider text-slate-700">
              Review Title
              <span className="ml-2 font-normal text-slate-400 capitalize">
                (Optional)
              </span>
            </label>

            <input
              type="text"
              value={reviewTitle}
              onChange={(e) => {
                setReviewTitle(e.target.value)
                setResult(null)
              }}
              maxLength={500}
              placeholder="e.g. Outstanding build quality and fast shipping!"
              className="w-full rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:bg-white focus:border-violet-500 focus:ring-4 focus:ring-violet-500/10"
            />
          </div>

          {/* Review Text */}
          <div>
            <label className="mb-2 block text-xs font-bold uppercase tracking-wider text-slate-700">
              Review Text
              <span className="ml-2 font-normal text-slate-400 capitalize">
                (Required if no image)
              </span>
            </label>

            <textarea
              value={reviewText}
              onChange={(e) => {
                setReviewText(e.target.value)
                setResult(null)
              }}
              rows={6}
              placeholder="Paste or write the customer's detailed product review here..."
              className="w-full resize-y rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:bg-white focus:border-violet-500 focus:ring-4 focus:ring-violet-500/10"
            />

            <p className="mt-1.5 text-right text-xs font-medium text-slate-400">
              {reviewText.length} characters
            </p>
          </div>

          {/* Image Upload */}
          <div>
            <label className="mb-2 block text-xs font-bold uppercase tracking-wider text-slate-700">
              Customer Review Image
              <span className="ml-2 font-normal text-slate-400 capitalize">
                (Optional)
              </span>
            </label>

            {!image ? (
              <label className="group flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-purple-200 bg-purple-50/30 px-6 py-9 transition hover:border-violet-500 hover:bg-violet-50/60">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-violet-100 text-violet-600 transition-transform group-hover:scale-110">
                  <Upload size={22} />
                </div>

                <p className="mt-3 text-sm font-semibold text-slate-800">
                  Click to upload review image
                </p>

                <p className="mt-1 text-xs text-slate-400">
                  JPEG, PNG or WebP · Up to 5 MB
                </p>

                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleImageChange}
                  className="hidden"
                />
              </label>
            ) : (
              <div className="relative rounded-2xl border border-purple-100 bg-purple-50/20 p-3">
                <img
                  src={imagePreview}
                  alt="Uploaded customer review"
                  className="mx-auto max-h-60 rounded-xl object-contain shadow-xs"
                />

                <button
                  type="button"
                  onClick={removeImage}
                  className="absolute right-4 top-4 rounded-full bg-slate-900/80 p-2 text-white shadow-md backdrop-blur-xs transition hover:bg-rose-600"
                  aria-label="Remove image"
                >
                  <X size={16} />
                </button>

                <p className="mt-2.5 truncate text-center text-xs font-semibold text-slate-600">
                  {image.name}
                </p>
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm font-medium text-rose-700">
              <AlertCircle size={18} className="mt-0.5 shrink-0 text-rose-600" />
              <span>{error}</span>
            </div>
          )}

          {/* Actions */}
          <div className="flex flex-wrap gap-3 pt-2">
            <button
              type="submit"
              disabled={loading}
              className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 px-6 py-3.5 text-sm font-bold text-white shadow-md shadow-violet-500/25 transition hover:from-violet-500 hover:to-fuchsia-500 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Running Neural Models...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Analyze Sentiment
                </>
              )}
            </button>

            <button
              type="button"
              onClick={resetForm}
              disabled={loading}
              className="rounded-xl border border-slate-200 bg-white px-5 py-3.5 text-sm font-semibold text-slate-700 shadow-xs transition hover:border-slate-300 hover:bg-slate-50"
            >
              Clear
            </button>
          </div>
        </form>

        {/* Prediction Result */}
        <div className="rounded-2xl border border-purple-100/90 bg-white p-6 sm:p-7 shadow-xs">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="text-violet-600" size={20} />
            <h2 className="text-lg font-bold text-slate-900 tracking-tight">
              Prediction Result
            </h2>
          </div>

          <p className="mt-0.5 text-xs text-slate-500 font-medium">
            Computed sentiment classification and probability tensor
          </p>

          {!result ? (
            <div className="flex min-h-[440px] flex-col items-center justify-center text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-purple-50 text-purple-400">
                <ImageIcon size={32} />
              </div>

              <h3 className="mt-4 font-bold text-slate-800">
                No prediction yet
              </h3>

              <p className="mt-1.5 max-w-xs text-xs text-slate-400">
                Provide review text, upload an image, or both and click "Analyze Sentiment" to compute predictions.
              </p>
            </div>
          ) : (
            <div className="mt-6 space-y-6">
              {/* Highlight card */}
              <div className="rounded-2xl border border-purple-100 bg-gradient-to-br from-purple-50/60 via-white to-pink-50/30 p-5 shadow-xs">
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Predicted Sentiment
                </p>

                <div className="mt-2.5 flex items-center justify-between">
                  <span
                    className={`inline-block rounded-xl border px-4 py-2 text-lg font-extrabold shadow-xs ${sentimentStyles[result.sentiment] || ''}`}
                  >
                    {result.sentiment}
                  </span>

                  <span className="rounded-lg bg-violet-100 px-3 py-1.5 text-xs font-bold text-violet-700">
                    Class {result.predicted_class}
                  </span>
                </div>

                <div className="mt-5 flex items-end justify-between">
                  <div>
                    <p className="text-xs font-medium text-slate-500">
                      Confidence Score
                    </p>
                    <p className="text-3xl font-extrabold text-slate-900 tracking-tight">
                      {(result.confidence * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>

                <div className="mt-3 h-2.5 overflow-hidden rounded-full bg-purple-100">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-violet-600 to-fuchsia-600 transition-all duration-500"
                    style={{
                      width: `${result.confidence * 100}%`,
                    }}
                  />
                </div>
              </div>

              {/* Class probabilities */}
              <div className="space-y-3.5">
                <h3 className="text-sm font-bold text-slate-900">
                  Class Probability Distribution
                </h3>

                {classNames.map((name) => {
                  const probability = result.probabilities[name] || 0

                  return (
                    <div key={name} className="space-y-1.5">
                      <div className="flex justify-between text-xs font-medium">
                        <span className="text-slate-600">{name}</span>
                        <span className="font-bold text-slate-900">
                          {(probability * 100).toFixed(2)}%
                        </span>
                      </div>

                      <div className="h-2 overflow-hidden rounded-full bg-purple-100">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-500"
                          style={{
                            width: `${probability * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Metadata tags */}
              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="rounded-xl border border-slate-100 bg-slate-50 p-3.5">
                  <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                    Model Used
                  </p>
                  <p className="mt-1 text-sm font-bold text-violet-700">
                    {result.model_used.replace('_', ' ')}
                  </p>
                </div>

                <div className="rounded-xl border border-slate-100 bg-slate-50 p-3.5">
                  <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                    Image Included
                  </p>
                  <p className="mt-1 text-sm font-bold text-slate-800">
                    {result.image_used ? 'Yes' : 'No'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Analyze
