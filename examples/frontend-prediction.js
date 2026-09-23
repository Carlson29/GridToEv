const DEFAULT_API_BASE_URL = "http://localhost:8000";

function requestHeaders(apiKey, includeJson = false) {
  return {
    ...(includeJson ? { "Content-Type": "application/json" } : {}),
    ...(apiKey ? { "X-API-Key": apiKey } : {}),
  };
}

export async function getDatasetInfo({
  apiBaseUrl = DEFAULT_API_BASE_URL,
  apiKey = "",
} = {}) {
  const response = await fetch(`${apiBaseUrl}/dataset/info`, {
    headers: requestHeaders(apiKey),
  });
  if (!response.ok) {
    throw new Error(`GridToEV dataset lookup failed: ${await response.text()}`);
  }
  return response.json();
}

export async function getLatestPredictions(
  flexibleLoadCapacityMw = 100,
  { apiBaseUrl = DEFAULT_API_BASE_URL, apiKey = "" } = {},
) {
  const query = new URLSearchParams({
    flexible_load_capacity_mw: String(flexibleLoadCapacityMw),
  });
  const response = await fetch(`${apiBaseUrl}/predict/latest?${query}`, {
    headers: requestHeaders(apiKey),
  });
  if (!response.ok) {
    throw new Error(`GridToEV prediction failed: ${await response.text()}`);
  }
  return response.json();
}
export async function predictHistoricalInterval({
  issueTimestampUtc,
  forecastHorizonMinutes,
  flexibleLoadCapacityMw = 100,
  apiBaseUrl = DEFAULT_API_BASE_URL,
  apiKey = "",
}) {
  const response = await fetch(`${apiBaseUrl}/predict/from-dataset`, {
    method: "POST",
    headers: requestHeaders(apiKey, true),
    body: JSON.stringify({
      issue_timestamp_utc: issueTimestampUtc,
      forecast_horizon_minutes: forecastHorizonMinutes,
      flexible_load_capacity_mw: flexibleLoadCapacityMw,
    }),
  });
  if (!response.ok) {
    throw new Error(`GridToEV prediction failed: ${await response.text()}`);
  }
  return response.json();
}

export async function predictHistoricalWindow({
  startTimestampUtc,
  durationHours = 2,
  forecastHorizonsMinutes = [30, 60],
  flexibleLoadCapacityMw = 100,
  apiBaseUrl = DEFAULT_API_BASE_URL,
  apiKey = "",
}) {
  const response = await fetch(`${apiBaseUrl}/predict/window/from-dataset`, {
    method: "POST",
    headers: requestHeaders(apiKey, true),
    body: JSON.stringify({
      start_timestamp_utc: startTimestampUtc,
      duration_hours: durationHours,
      forecast_horizons_minutes: forecastHorizonsMinutes,
      flexible_load_capacity_mw: flexibleLoadCapacityMw,
    }),
  });
  if (!response.ok) {
    throw new Error(`GridToEV window prediction failed: ${await response.text()}`);
  }
  return response.json();
}
