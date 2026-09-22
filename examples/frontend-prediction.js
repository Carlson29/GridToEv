const API_BASE_URL = "http://localhost:8000";

export async function getLatestPredictions(flexibleLoadCapacityMw = 100) {
  const query = new URLSearchParams({
    flexible_load_capacity_mw: String(flexibleLoadCapacityMw),
  });
  const response = await fetch(`${API_BASE_URL}/predict/latest?${query}`);
  if (!response.ok) {
    throw new Error(`GridToEV prediction failed: ${await response.text()}`);
  }
  return response.json();
}
export async function predictHistoricalInterval({
  issueTimestampUtc,
  forecastHorizonMinutes,
  flexibleLoadCapacityMw = 100,
}) {
  const response = await fetch(`${API_BASE_URL}/predict/from-dataset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
