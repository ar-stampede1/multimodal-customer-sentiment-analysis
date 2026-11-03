
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
})

// Dashboard statistics
export const getDashboardStatistics = async () => {
  const response = await api.get('/statistics/dashboard')
  return response.data
}

// Prediction history
export const getPredictionHistory = async (limit = 10, offset = 0) => {
  const response = await api.get('/predictions/history', {
    params: { limit, offset },
  })

  return response.data
}

// Sentiment prediction
export const predictSentiment = async ({
  reviewTitle,
  reviewText,
  image,
}) => {
  const formData = new FormData()

  formData.append('review_title', reviewTitle || '')
  formData.append('review_text', reviewText || '')

  if (image) {
    formData.append('image', image)
  }

  const response = await api.post(
    '/predictions/predict',
    formData
  )

  return response.data
}

export default api
