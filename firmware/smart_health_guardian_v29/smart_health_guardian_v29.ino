#include <Wire.h>
#include "MAX30105.h"
#include <Anastasia-Health-Project_inferencing.h>

MAX30105 particleSensor;

#define DSP_INPUT_FRAME_SIZE EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE
float features[DSP_INPUT_FRAME_SIZE];


#define DEBUG_MODE 0

// 1 = also print every raw sample ("R,t_ms,ir,red") and every detected beat ("B,t_ms") for offline
// analysis of the beat detector on real signals. Keep 0 for normal use.
#define RAW_STREAM 0


#define WARMUP_MS 15000UL


const float TEMP_OFFSET     = 4.3f;
const bool  TEMP_VALID      = false;  // single-point estimate: NOT validated -> Fever is shown as Uncertain
const float SPO2_CORRECTION = 3.0f;   // V29 session vs Vicko: raw 95.8% vs oximeter ~98.6% (8 readings)


const byte  RATE_SIZE      = 4;
byte        rates[RATE_SIZE];
byte        rateSpot       = 0;
double               lastBeatTime   = 0.0;   // ms on the sensor clock, sub-sample resolution
bool                 have_last_beat = false;
unsigned long        sample_clock_ms = 0;
double               dec_red = 0.0, dec_ir = 0.0;
int                  dec_n = 0;
bool                 dec_motion = false;   // any motion inside the current SpO2 decimation group
unsigned long        lastDelta      = 0;
float       beatsPerMinute = 0.0f;
float       avgBPM         = 0.0f;
const float BPM_ALPHA      = 0.10f;


double avered        = 0.0;
double aveir         = 0.0;
double sumirrms      = 0.0;
double sumredrms     = 0.0;
int    spo2_sample_i = 0;
int    spo2_motion_n = 0;      // samples of the current SpO2 window rejected as motion
double aveir_prev_window = 0.0;
const  int    SPO2_SAMPLE_NUM = 50;   // 50 samples @25 Hz = 2 s window
const  int    SPO2_MAX_MOTION = SPO2_SAMPLE_NUM / 10;  // window rejected if >10% of it is motion
const  int    SPO2_DECIM      = 4;    // 100 Hz -> 25 Hz for SpO2
const  unsigned long SAMPLE_PERIOD_MS = 10; // 100 Hz sample clock
const  uint16_t LIB_STORAGE_SIZE = 4;       // STORAGE_SIZE of the SparkFun MAX3010x library
unsigned long samples_lost_total = 0;       // samples dropped by the library buffer since boot
const  float  DRIFT_THRESHOLD = 0.01f;
int    spo2_windows_seen = 0;   // first windows after finger-on are discarded
byte   bpm_count = 0;           // valid beat intervals since finger-on
double ESpO2         = 0.0;
const  double FSpO2  = 0.85;
const  double frate  = 0.95;   // tau ~ 20 samples @25 Hz = 0.8 s
bool   spo2_valid    = false;
double last_R        = 0.0;    // last ratio-of-ratios before clamping (debug)


const int MAX_RR_LEN    = 20;
const int MIN_HRV_PAIRS = 5;

int   rr_history[MAX_RR_LEN];   // 0 = gap (rejected beat): pairs across a gap are skipped
int   rr_index   = 0;
int   rr_count   = 0;

float avgHRV     = 0.0f;
float pnn50      = 0.0f;
bool  hrv_valid  = false;
const float HRV_ALPHA = 0.10f;


float signalQuality  = 0.0f;
float dc_slow_ema    = 0.0f;
const float DC_ALPHA = 0.05f;

float ac_envelope    = 0.0f;
const float AC_ENV_ALPHA = 0.02f;   // tau ~2.5 s @20 Hz: follows perfusion, not the phase of each beat
                                    // (0.10 made SigQ jump 15%..87% from second to second)
const float MOTION_THRESHOLD     = 0.02f;   // minimum motion limit (relative deviation from DC)
const float MOTION_MAX_THRESHOLD = 0.15f;   // upper cap, so large movements are always caught
const float MOTION_ENV_FACTOR    = 3.0f;    // motion = deviation > 3x the typical pulse amplitude
bool  motion_since_last_beat = false;      // any motion sample since the last accepted beat
float last_pi        = 0.0f;               // perfusion index (%) from the last SigQ update
const long  FINGER_IR_MIN        = 7000;    // IR counts above which a finger is present
const float RR_TOLERANCE         = 0.18f;   // same as V28: at 25% the beat-timing jitter of checkForBeat gave
                                            // RMSSD 100-129 ms at rest (82/111 false Arrhythmia)
const float HRV_MIN_SIGQ         = 0.55f;   // a beat counts for HRV only with SigQ >= this and no motion
                                            // (0.50 with the fast SigQ gave RMSSD ~164 ms -> false Arrhythmia,
                                            //  0.65 rejected ~3 of 4 beats -> HRV never valid)
const float MIN_CONFIDENCE       = 0.60f;   // below this the result is shown as Uncertain
const float ARRHYTHMIA_MIN_CONF  = 0.80f;   // Arrhythmia needs high model confidence...
const float ARRHYTHMIA_MIN_SIGQ  = 0.60f;   // ...and good signal quality
const float TEMP_MIN_C           = 35.0f;   // plausible body-temperature window
const float TEMP_MAX_C           = 42.0f;
const float TEMP_NEUTRAL_C       = 36.7f;   // fed to the model while TEMP_VALID is false (mean of the normal classes)
const float POOR_SIGNAL_SIGQ     = 0.30f;   // below this the result is Poor_signal by rule
const float ANOMALY_MAX          = 0.30f;   // K-means anomaly score above this -> Uncertain (training data: max 0.13)
const float HYPOXIA_MAX_SPO2     = 93.0f;   // Hypoxia is shown only with a measured SpO2 below this
                                            // (device test: model said Hypoxia at SpO2 98.6% - it under-weights SpO2)
const float POOR_SIGNAL_MODEL_SIGQ = 0.50f; // model "Poor_signal" with SigQ above this is shown as Uncertain

// Model inputs per row: BPM, SpO2, Temp, RMSSD, pNN50 [, SignalQuality].
// Works with both libraries: 6 axes (SigQ as input) or 5 axes (SigQ removed, synthetic SigQ ~1.0
// made every real reading with SigQ 0.6-0.85 look "not Relaxed").
const int MODEL_AXES = EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
float body_temp  = 0.0f;
float current_temp_skin = 0.0f;               // last die-temperature reading (refreshed every TEMP_PERIOD_MS)
int   window_rows = 0;                        // rows currently in the sliding window
const unsigned long ROW_PERIOD_MS   = 29;     // ~34 Hz, matches the Edge Impulse impulse (34 Hz)
const unsigned long INFER_STRIDE_MS = 1000;   // classify every 1 s -> same stride as the impulse
const unsigned long TEMP_PERIOD_MS  = 1000;   // temperature changes slowly: read it once per second
bool  temp_sane  = false;
long  irValue    = 0;
long  redValue   = 0;

#if DEBUG_MODE
// counted only while a finger is present, reset after every report (= per ~1 s)
unsigned long dbg_finger_samples = 0;   // raw samples with finger present
unsigned long dbg_motion_samples = 0;   // ...of which flagged as motion
unsigned long dbg_clean_beats    = 0;   // RR intervals stored for HRV
unsigned long dbg_gap_beats      = 0;   // RR intervals stored as gap (0)
unsigned long dbg_spo2_rejected  = 0;   // SpO2 windows rejected for motion
const int     DBG_RR_LEN         = 10;
int           dbg_raw_rr[DBG_RR_LEN];   // last beat intervals before any HRV filtering
int           dbg_raw_rr_i       = 0;

void printRecentRR() {
  Serial.print(" [DBG] RR raw (ms):");
  for (int k = 0; k < DBG_RR_LEN; k++) {
    Serial.print(' ');
    Serial.print(dbg_raw_rr[(dbg_raw_rr_i + k) % DBG_RR_LEN]);
  }
  Serial.print("  | stored (0=gap):");
  int n = min(rr_count, DBG_RR_LEN);
  for (int k = n - 1; k >= 0; k--) {
    Serial.print(' ');
    Serial.print(rr_history[(rr_index - 1 - k + MAX_RR_LEN) % MAX_RR_LEN]);
  }
  Serial.println();
}
#endif


// Motion = deviation from DC much larger than the user's own pulse amplitude.
// A fixed 2% limit flagged strong, clean pulses (PI > ~1.5%) as motion.
bool isMotion(float abs_ac, float dc) {
  if (dc <= 1000.0f) return true;   // DC not ready -> treat as unusable
  float limit = constrain(MOTION_ENV_FACTOR * ac_envelope / dc,
                          MOTION_THRESHOLD, MOTION_MAX_THRESHOLD);
  return (abs_ac / dc) > limit;
}


// ---- beat detector (replaces SparkFun checkForBeat) ----
// checkForBeat gave beat-timing jitter of ~+/-50 ms (pNN50 ~80% at rest), too much for HRV.
// Here: band-pass 0.5-4 Hz; a beat is confirmed when the pulse rises from its last trough by
// 50% of the typical pulse height and then falls clearly after the peak (robust to noise and to
// respiratory baseline wander). Its time is the steepest point of the systolic upstroke (much
// sharper than the rounded peak), refined with parabolic interpolation (<10 ms resolution).
// Simulation (synthetic PPG, true RMSSD 38 ms, with respiratory wander and amplitude modulation):
// RMSSD after the 18% filter 38-56 ms for noise 0-20% of the pulse.
const float  BEAT_HP_ALPHA       = 0.031f;  // DC removal, fc ~0.5 Hz @100 Hz
const double BEAT_MIN_SPACING_MS = 250.0;   // hard refractory (240 bpm)
const int    BEAT_SETTLE_SAMPLES = 150;     // 1.5 s for the filters to settle after finger-on
const float  BEAT_RISE_FRAC      = 0.5f;    // rise from the trough needed to confirm an upstroke
const float  BEAT_DROP_FRAC      = 0.2f;    // fall from the peak needed to end the upstroke
const int    SLOPE_HIST          = 50;      // 500 ms of slope history
float  lp_b0, lp_b1, lp_b2, lp_a1, lp_a2;   // 2nd-order Butterworth low-pass, 4 Hz
float  bp_dc = 0.0f;
float  lp_x1 = 0.0f, lp_x2 = 0.0f, lp_y1 = 0.0f, lp_y2 = 0.0f;
float  bp_prev = 0.0f;
float  slope_hist[SLOPE_HIST];
long   slope_n = 0;                         // number of slopes written to slope_hist
float  pulse_amp = 0.0f;                    // typical trough-to-peak height of the filtered pulse
float  trough_y = 0.0f, peak_y = 0.0f;
float  settle_min = 0.0f, settle_max = 0.0f;
const double BEAT_MAX_REFRACT_MS = 450.0;   // refractory never longer than this (allows up to ~130 bpm)
double rr_est = 0.0;                        // typical beat interval (adaptive refractory, BPM check)
int    rr_est_misses = 0;                   // inconsistent intervals in a row
double last_detect_t = -1e9;
int    beat_settle = 0;
byte   beat_state = 0;                      // 0 armed (following the trough), 1 on the upstroke
long   best_n = 0;                          // index of the steepest slope on the current upstroke
float  best_slope = 0.0f;

void initBeatFilter() {
  const float fs = 1000.0f / SAMPLE_PERIOD_MS;
  const float K  = tanf(PI * 4.0f / fs);
  const float Q  = 0.7071f;
  float norm = 1.0f / (1.0f + K / Q + K * K);
  lp_b0 = K * K * norm;
  lp_b1 = 2.0f * lp_b0;
  lp_b2 = lp_b0;
  lp_a1 = 2.0f * (K * K - 1.0f) * norm;
  lp_a2 = (1.0f - K / Q + K * K) * norm;
}

void resetBeatDetector() {
  bp_dc = 0.0f;
  lp_x1 = lp_x2 = lp_y1 = lp_y2 = 0.0f;
  bp_prev = 0.0f;
  for (int k = 0; k < SLOPE_HIST; k++) slope_hist[k] = 0.0f;
  slope_n = 0;
  pulse_amp = 0.0f;
  trough_y = peak_y = 0.0f;
  settle_min = 1e30f; settle_max = -1e30f;
  rr_est = 0.0;
  rr_est_misses = 0;
  last_detect_t = -1e9;
  beat_settle = 0;
  beat_state = 0;
}

static inline float slopeAt(long n) { return slope_hist[n % SLOPE_HIST]; }

// Feed one IR sample taken at t_ms. Returns true when a beat is found and sets beat_t.
bool detectBeat(float ir, double t_ms, double &beat_t) {
  if (bp_dc == 0.0f) bp_dc = ir;
  bp_dc += BEAT_HP_ALPHA * (ir - bp_dc);
  float hp = bp_dc - ir;                      // IR falls when blood volume rises -> invert

  float y = lp_b0 * hp + lp_b1 * lp_x1 + lp_b2 * lp_x2 - lp_a1 * lp_y1 - lp_a2 * lp_y2;
  lp_x2 = lp_x1; lp_x1 = hp;
  lp_y2 = lp_y1; lp_y1 = y;

  float slope = y - bp_prev;
  bp_prev = y;
  long  n_now = slope_n;                      // index of this slope
  slope_hist[n_now % SLOPE_HIST] = slope;
  slope_n++;

  // settle: the first pulse-height estimate is the signal range over 1.5 s
  if (beat_settle < BEAT_SETTLE_SAMPLES) {
    beat_settle++;
    if (y < settle_min) settle_min = y;
    if (y > settle_max) settle_max = y;
    if (beat_settle == BEAT_SETTLE_SAMPLES) { pulse_amp = settle_max - settle_min; trough_y = y; }
    return false;
  }

  pulse_amp *= 0.999f;                        // slow decay, recovers after a weaker signal
  bool found = false;

  if (beat_state == 0) {
    if (y < trough_y) trough_y = y;
    // capped at 450 ms: an uncapped 0.6*rr_est could lock the detector on every 2nd beat
    // (seen on the device at ~100 bpm: RR ~1100 ms for 45 s)
    double refractory = max(BEAT_MIN_SPACING_MS, min(0.6 * rr_est, BEAT_MAX_REFRACT_MS));
    if (y - trough_y >= BEAT_RISE_FRAC * pulse_amp && (t_ms - last_detect_t) >= refractory) {
      // upstroke confirmed: look back over the positive-slope run for its steepest point
      best_slope = -1e9f;
      best_n     = n_now;
      for (int k = 0; k < SLOPE_HIST - 2 && k <= n_now; k++) {
        long n = n_now - k;
        if (slopeAt(n) <= 0.0f) break;
        if (slopeAt(n) > best_slope) { best_slope = slopeAt(n); best_n = n; }
      }
      peak_y     = y;
      beat_state = 1;
    }
  } else {                                    // beat_state == 1: on the upstroke / at the peak
    if (y > peak_y) peak_y = y;
    if (slope > best_slope && y >= peak_y) { best_slope = slope; best_n = n_now; }
    if (y < peak_y - BEAT_DROP_FRAC * pulse_amp) {   // clearly past the systolic peak
      float a = slopeAt(best_n - 1), b = slopeAt(best_n), c = slopeAt(best_n + 1);
      float den    = a - 2.0f * b + c;
      float offset = (den != 0.0f) ? 0.5f * (a - c) / den : 0.0f;
      offset = constrain(offset, -0.5f, 0.5f);
      beat_t = t_ms - (double)(n_now - best_n) * SAMPLE_PERIOD_MS + offset * SAMPLE_PERIOD_MS;

      // pulse height: capped against motion spikes, floored so noise cannot collapse it
      float h = peak_y - trough_y;
      pulse_amp = 0.8f * pulse_amp + 0.2f * max(min(h, 2.0f * pulse_amp), 0.5f * pulse_amp);

      // typical interval: only consistent intervals update it; if the rhythm really changed
      // (5 inconsistent intervals in a row) it is re-learned from scratch
      double iv = beat_t - last_detect_t;
      if (iv > 300.0 && iv < 1500.0 && (rr_est == 0.0 || (iv > 0.7 * rr_est && iv < 1.3 * rr_est))) {
        rr_est = (rr_est == 0.0) ? iv : 0.8 * rr_est + 0.2 * iv;
        rr_est_misses = 0;
      } else if (++rr_est_misses >= 5) {
        rr_est = 0.0;
        rr_est_misses = 0;
      }

      last_detect_t = beat_t;
      trough_y      = y;
      beat_state    = 0;
      found         = true;
    }
  }
  return found;
}


void updateSignalQuality(long ir_sample) {
  static int dc_settle_count = 0;


  if (dc_slow_ema == 0.0f) {
    dc_slow_ema     = (float)ir_sample;
    dc_settle_count = 0;
    ac_envelope     = 0.0f;
    signalQuality   = 0.0f;
    return;
  }

  dc_slow_ema = DC_ALPHA * (float)ir_sample + (1.0f - DC_ALPHA) * dc_slow_ema;
  dc_settle_count++;

  if (dc_slow_ema < 1000.0f) { signalQuality = 0.0f; return; }


  const int DC_SETTLE_SAMPLES = 60;
  if (dc_settle_count < DC_SETTLE_SAMPLES) { signalQuality = 0.0f; return; }


  float abs_ac = fabsf((float)ir_sample - dc_slow_ema);


  bool is_motion = isMotion(abs_ac, dc_slow_ema);

  if (!is_motion) {
    ac_envelope = AC_ENV_ALPHA * abs_ac + (1.0f - AC_ENV_ALPHA) * ac_envelope;
  }


  if (dc_settle_count < DC_SETTLE_SAMPLES + (int)(1.5f / AC_ENV_ALPHA)) {
    signalQuality = 0.0f;
    return;
  }


  float pi = (ac_envelope / dc_slow_ema) * 100.0f;
  last_pi = pi;


  if      (pi < 0.15f) signalQuality = 0.0f;
  else if (pi < 0.30f) signalQuality = (pi - 0.15f) / 0.15f * 0.5f;
  else if (pi < 0.70f) signalQuality = 0.5f + (pi - 0.30f) / 0.40f * 0.3f;
  else if (pi < 4.0f)  signalQuality = 0.8f + (pi - 0.70f) / 3.30f * 0.2f;
  else if (pi < 12.0f) signalQuality = 1.0f - (pi - 4.0f)  / 8.0f  * 0.5f;
  else                 signalQuality = max(0.0f, 0.5f - (pi - 12.0f) * 0.04f);

  signalQuality = constrain(signalQuality, 0.0f, 1.0f);
}


bool calculateHRV() {
  if (rr_count < MIN_HRV_PAIRS + 1) {
    hrv_valid = false;
    return false;
  }

  double sum_sq_diff = 0.0;
  int    valid_pairs = 0;
  int    nn50_count  = 0;

  int samples_to_check = min(rr_count - 1, MAX_RR_LEN - 1);

  for (int k = 0; k < samples_to_check; k++) {
    int idx_now  = (rr_index - 1 - k + MAX_RR_LEN) % MAX_RR_LEN;
    int idx_prev = (rr_index - 2 - k + MAX_RR_LEN) % MAX_RR_LEN;

    int RR_now  = rr_history[idx_now];
    int RR_prev = rr_history[idx_prev];

    if (RR_prev == 0 || RR_now == 0) continue;   // gap: the two intervals are not successive

    int diff = abs(RR_now - RR_prev);
    if (diff > (int)(RR_TOLERANCE * RR_prev)) continue;

    sum_sq_diff += (double)diff * (double)diff;
    valid_pairs++;
    if (diff > 50) nn50_count++;
  }

  if (valid_pairs < MIN_HRV_PAIRS) {
    hrv_valid = false;
    return false;
  }

  float raw_rmssd = (float)sqrt(sum_sq_diff / valid_pairs);
  float raw_pnn50 = ((float)nn50_count / valid_pairs) * 100.0f;

  if (avgHRV == 0.0f) avgHRV = raw_rmssd;
  else                avgHRV = HRV_ALPHA * raw_rmssd + (1.0f - HRV_ALPHA) * avgHRV;

  pnn50     = raw_pnn50;
  hrv_valid = true;
  return true;
}


void setup() {
  // Large TX buffer: printing a report (~1 KB) must not block loop(). The SparkFun library keeps only
  // STORAGE_SIZE = 4 samples, so a loop() stall > 40 ms silently lost samples (measured: 92-95 instead
  // of 100 samples/s while reports were printed -> intervals too short, BPM ~5% high, HRV inflated).
  Serial.setTxBufferSize(4096);
  Serial.begin(115200);
  Wire.begin(18, 19);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30105 not found. Check wiring/power.");
    while (1);
  }

  particleSensor.setup(0x1F, 4, 2, 400, 411, 16384);  // sampleAverage=4 -> 100 Hz effective
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeGreen(0);
  particleSensor.enableDIETEMPRDY();

  for (int k = 0; k < MAX_RR_LEN; k++) rr_history[k] = 0;
  ac_envelope = 0.0f;
  initBeatFilter();
  resetBeatDetector();
#if DEBUG_MODE
  for (int k = 0; k < DBG_RR_LEN; k++) dbg_raw_rr[k] = 0;
#endif

  Serial.println("\n-------------------------------------------");
  Serial.println(" SMART HEALTH GUARDIAN - V29.0            ");
  Serial.println("-------------------------------------------");
  Serial.println(" Research prototype - not a medical device.");
#if DEBUG_MODE
  Serial.println(" DEBUG MODE: ON");
#endif
  Serial.print(" Warmup: "); Serial.print(WARMUP_MS / 1000UL);
  Serial.println("s after finger detected.");
  Serial.println();
}

void classifyAndReport() {
  Serial.println("\nRunning AI classifier...");

  signal_t signal;
  int err = numpy::signal_from_buffer(features, DSP_INPUT_FRAME_SIZE, &signal);
  if (err != 0) { Serial.println("Buffer error - skipping."); return; }

  ei_impulse_result_t result = { 0 };
  unsigned long t0 = micros();
  run_classifier(&signal, &result, false);
  Serial.printf("Inference: %lu us\n", micros() - t0);

  int   best_idx  = -1;
  float max_score = 0.0f;
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    if (result.classification[ix].value > max_score) {
      max_score = result.classification[ix].value;
      best_idx  = (int)ix;
    }
  }

  const char* ai_label = (best_idx >= 0) ? result.classification[best_idx].label : "Unknown";


  float final_spo2 = constrain((float)ESpO2 + SPO2_CORRECTION, 70.0f, 100.0f);

#if EI_CLASSIFIER_HAS_ANOMALY
  float anomaly = result.anomaly;     // K-means distance to the nearest training cluster
#else
  float anomaly = 0.0f;
#endif

  Serial.println("\n+===========================================+");

  Serial.print(" BPM:   ");
  if (avgBPM > 0) Serial.println(avgBPM, 1);
  else            Serial.println("--");

  Serial.print(" SpO2:  ");
  if (spo2_valid) {
    Serial.print(final_spo2, 1); Serial.println("%");
    if (SPO2_CORRECTION == 0.0f)
      Serial.println("        [reflective sensor - may read 2-4% low]");
    else
      Serial.println("        [reflective sensor + correction applied]");
  } else {
    Serial.println("-- (stabilising)");
  }

  Serial.print(" Temp:  ");
  if (temp_sane) {
    Serial.print(body_temp, 1); Serial.print(" C");
    if (TEMP_VALID) Serial.println("  [estimate, single-point calibration]");
    else            Serial.println("  [not validated]");
  } else {
    Serial.print(current_temp_skin, 1);
    Serial.println(" C (skin only - TEMP_OFFSET needs adjustment)");
  }

  Serial.print(" HRV:   ");
  if (hrv_valid) {
    Serial.print(avgHRV, 1);
    Serial.print(" ms RMSSD  pNN50: ");
    Serial.print(pnn50, 1); Serial.println("%");
  } else {
    Serial.println("-- (collecting beats)");
  }

  Serial.print(" SigQ:  ");
  Serial.print(signalQuality * 100.0f, 0); Serial.print("%  ");
  if      (signalQuality >= 0.8f) Serial.println("[Excellent]");
  else if (signalQuality >= 0.5f) Serial.println("[Good]");
  else if (signalQuality >= 0.3f) Serial.println("[Fair]");
  else if (signalQuality >= 0.1f) Serial.println("[Poor - adjust finger]");
  else                            Serial.println("[No signal]");

#if DEBUG_MODE
  Serial.printf(" [DBG] PI: %.2f%%  motion: %lu/%lu samples  RR clean/gap: %lu/%lu  SpO2 windows rejected: %lu  R: %.3f  raw SpO2: %.1f%%  samples lost (total): %lu\n",
                last_pi, dbg_motion_samples, dbg_finger_samples, dbg_clean_beats, dbg_gap_beats, dbg_spo2_rejected,
                last_R, ESpO2, samples_lost_total);
  printRecentRR();
  dbg_finger_samples = 0; dbg_motion_samples = 0;
  dbg_clean_beats = 0; dbg_gap_beats = 0; dbg_spo2_rejected = 0;
#endif

  Serial.println();

  if (avgBPM > 40) {

    // Display rules (the model output itself is never modified):
    //  - Arrhythmia needs SigQ >= ARRHYTHMIA_MIN_SIGQ and confidence >= ARRHYTHMIA_MIN_CONF
    //  - Fever is not reported while the temperature estimate is not validated
    //  - any result below MIN_CONFIDENCE is shown as Uncertain
    //  - SigQ below POOR_SIGNAL_SIGQ is reported as Poor_signal by rule (SigQ is not a model input)
    //  - an input far from all training data (K-means anomaly score) is shown as Uncertain
    //  - Hypoxia needs a measured SpO2 below HYPOXIA_MAX_SPO2
    //  - a model "Poor_signal" while SigQ is good is shown as Uncertain
    const char* display_label = ai_label;
    const char* reason        = nullptr;
    if (signalQuality < POOR_SIGNAL_SIGQ) {
      display_label = "Poor_signal";
      reason        = "signal quality below 30%";
    } else if (anomaly > ANOMALY_MAX) {
      display_label = "Uncertain";
      reason        = "outside the training data (anomaly score)";
    } else if (strcmp(ai_label, "Hypoxia") == 0 && (!spo2_valid || final_spo2 >= HYPOXIA_MAX_SPO2)) {
      display_label = "Uncertain";
      reason        = "Hypoxia needs a measured SpO2 below 93%";
    } else if (strcmp(ai_label, "Poor_signal") == 0 && signalQuality >= POOR_SIGNAL_MODEL_SIGQ) {
      display_label = "Uncertain";
      reason        = "model reports poor signal but signal quality is good";
    } else if (strcmp(ai_label, "Arrhythmia") == 0 &&
        (signalQuality < ARRHYTHMIA_MIN_SIGQ || max_score < ARRHYTHMIA_MIN_CONF)) {
      display_label = "Uncertain";
      reason        = "Arrhythmia needs good signal and high confidence";
    } else if (strcmp(ai_label, "Fever") == 0 && !TEMP_VALID) {
      display_label = "Uncertain";
      reason        = "temperature not validated";
    } else if (max_score < MIN_CONFIDENCE) {
      display_label = "Uncertain";
      reason        = "low model confidence";
    }

    Serial.print(" DIAGNOSIS:  "); Serial.println(display_label);
    Serial.print(" Confidence: "); Serial.print(max_score * 100.0f, 1); Serial.println("%");
#if EI_CLASSIFIER_HAS_ANOMALY
    Serial.print(" Anomaly:    "); Serial.print(anomaly, 2);
    Serial.println(anomaly > ANOMALY_MAX ? "  [outside training data]" : "");
#endif
    if (reason) { Serial.print(" [Suppressed: "); Serial.print(reason); Serial.println("]"); }

    if      (strcmp(display_label, "Active")      == 0) Serial.println(" > Active / elevated state.");
    else if (strcmp(display_label, "Relaxed")     == 0) Serial.println(" > Resting state.");
    else if (strcmp(display_label, "Stressed")    == 0) Serial.println(" > Elevated stress indicators.");
    else if (strcmp(display_label, "Arrhythmia")  == 0) Serial.println(" > Possible rhythm irregularity - not a medical diagnosis.");
    else if (strcmp(display_label, "Fever")       == 0) Serial.println(" > Elevated temperature estimate (sensor drift +/-1-2 C).");
    else if (strcmp(display_label, "Hypoxia")     == 0) Serial.println(" > Low SpO2 indicator.");
    else if (strcmp(display_label, "Poor_signal") == 0) Serial.println(" > Poor signal - adjust finger.");
    else if (strcmp(display_label, "Uncertain")   == 0) Serial.println(" > Improve signal quality / hold still.");
    else                                                Serial.println(" > Unclassified.");
  } else {
    Serial.println(" DIAGNOSIS: Calibrating - keep finger still.");
  }

  Serial.println("+===========================================+\n");
}

// ----------------------------------------------------------
void loop() {
  static unsigned long finger_detected_time = 0;
  static unsigned long last_warmup_print    = 0;
  static unsigned long last_msg_print       = 0;
  static unsigned long last_row_t           = 0;
  static unsigned long last_temp_t          = 0;
  static unsigned long last_infer_t         = 0;
  const int WINDOW_ROWS = DSP_INPUT_FRAME_SIZE / MODEL_AXES;

    // check() reads every sample waiting in the sensor FIFO but the library stores only the newest
    // LIB_STORAGE_SIZE of them: if more arrived, the oldest were dropped. Keep the sample clock right
    // by skipping their time, so beat intervals are not shortened.
    uint16_t n_new = particleSensor.check();
    if (n_new > LIB_STORAGE_SIZE) {
      unsigned long lost = n_new - LIB_STORAGE_SIZE;
      sample_clock_ms += lost * SAMPLE_PERIOD_MS;
      samples_lost_total += lost;
    }

    while (particleSensor.available()) {
      long fred_raw = particleSensor.getFIFORed();
      long fir_raw  = particleSensor.getFIFOIR();
      sample_clock_ms += SAMPLE_PERIOD_MS;   // jitter-free time base (sensor clock)


      if (fred_raw == 0 || fir_raw == 0) {
        particleSensor.nextSample();
        continue;
      }

      double fred = (double)fred_raw;
      double fir  = (double)fir_raw;

      redValue = fred_raw;
      irValue  = fir_raw;
#if RAW_STREAM
      Serial.printf("R,%lu,%ld,%ld\n", sample_clock_ms, fir_raw, fred_raw);
#endif


      bool motion_ok = !isMotion(fabsf((float)fir_raw - dc_slow_ema), dc_slow_ema);
      if (!motion_ok) {
        motion_since_last_beat = true;
        dec_motion = true;
      }
#if DEBUG_MODE
      if (fir_raw > FINGER_IR_MIN) {
        dbg_finger_samples++;
        if (!motion_ok) dbg_motion_samples++;
      }
#endif


      // SpO2 stream: average groups of SPO2_DECIM samples -> 25 Hz
      dec_red += fred; dec_ir += fir; dec_n++;
      if (dec_n >= SPO2_DECIM) {
        double fr = dec_red / dec_n;
        double fi = dec_ir  / dec_n;
        bool   fm = dec_motion;
        dec_red = 0.0; dec_ir = 0.0; dec_n = 0; dec_motion = false;

        if (avered == 0.0) { avered = fr; aveir = fi; }
        avered = avered * frate + fr * (1.0 - frate);
        aveir  = aveir  * frate + fi * (1.0 - frate);

        // motion samples are left out of the RMS sums (R uses the same samples
        // for both channels), the window is only rejected if too many are motion
        if (fm) {
          spo2_motion_n++;
        } else {
          sumredrms += (fr - avered) * (fr - avered);
          sumirrms  += (fi - aveir)  * (fi - aveir);
        }
        spo2_sample_i++;

        if (spo2_sample_i >= SPO2_SAMPLE_NUM) {
          bool drift_ok = true;
          if (aveir_prev_window > 0.0) {
            drift_ok = (fabs(aveir - aveir_prev_window) / aveir_prev_window < DRIFT_THRESHOLD);
          }
          aveir_prev_window = aveir;

          bool window_clean = (spo2_motion_n <= SPO2_MAX_MOTION);
#if DEBUG_MODE
          if (!window_clean) dbg_spo2_rejected++;
#endif

          spo2_windows_seen++;
          if (window_clean && drift_ok && spo2_windows_seen > 2 && avered > 0 && aveir > 0 && sumirrms > 0) {
            double R = (sqrt(sumredrms) / avered) / (sqrt(sumirrms) / aveir);
            last_R = R;
            R = constrain(R, 0.4, 1.8);
            double SpO2 = 104.0 - 17.0 * R;
            SpO2 = constrain(SpO2, 70.0, 100.0);

            if (ESpO2 == 0.0) ESpO2 = SpO2;
            else              ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2;

            spo2_valid = (ESpO2 >= 70.0);
          }
          sumredrms = 0.0; sumirrms = 0.0; spo2_sample_i = 0; spo2_motion_n = 0;
        }
      }

      { static int sig_skip = 0;
        if (++sig_skip >= 5) { sig_skip = 0; updateSignalQuality(irValue); } }


      if (irValue > FINGER_IR_MIN) {
        if (finger_detected_time == 0) finger_detected_time = millis();

        double beat_t;
        if (detectBeat((float)fir_raw, (double)sample_clock_ms, beat_t)) {
#if RAW_STREAM
          Serial.printf("B,%.1f\n", beat_t);
#endif
          if (!have_last_beat) {
            have_last_beat = true;
            lastBeatTime   = beat_t;
            motion_since_last_beat = false;
          } else {
            unsigned long delta = (unsigned long)(beat_t - lastBeatTime + 0.5);

            // adaptive refractory: rejects the dicrotic notch at rest, allows tachycardia
            unsigned long min_gap = (lastDelta > 0) ? 300UL : 400UL;
            if (lastDelta > 0) {
              unsigned long adaptive = (unsigned long)(0.6f * lastDelta);
              if (adaptive > min_gap) min_gap = adaptive;
            }

            if (delta >= min_gap) {
              lastBeatTime   = beat_t;
              beatsPerMinute = 60000.0f / (float)delta;
#if DEBUG_MODE
              dbg_raw_rr[dbg_raw_rr_i] = (int)delta;
              dbg_raw_rr_i = (dbg_raw_rr_i + 1) % DBG_RR_LEN;
#endif

              // a missed beat gives ~2x the usual interval: keep it out of the BPM average
              bool bpm_interval_ok = (rr_est == 0.0) ||
                                     ((double)delta > 0.7 * rr_est && (double)delta < 1.4 * rr_est);
              if (beatsPerMinute > 40 && beatsPerMinute < 200 && bpm_interval_ok) {
                rates[rateSpot++] = (byte)beatsPerMinute;
                rateSpot %= RATE_SIZE;
                if (bpm_count < RATE_SIZE) bpm_count++;

                if (bpm_count >= RATE_SIZE) {
                  int localAvg = 0;
                  for (byte x = 0; x < RATE_SIZE; x++) localAvg += rates[x];
                  localAvg /= RATE_SIZE;

                  if (avgBPM == 0.0f) avgBPM = (float)localAvg;
                  else avgBPM = BPM_ALPHA * localAvg + (1.0f - BPM_ALPHA) * avgBPM;
                }
              }

              // A beat is rejected for HRV because the signal was bad while it was
              // measured (motion / low SigQ), not because it differs from the previous
              // one - otherwise a real irregular rhythm could never be seen.
              // Rejected beats are stored as 0 so that RMSSD only uses successive intervals.
              bool in_range   = (delta > 300 && delta < 1500);
              bool beat_clean = (signalQuality >= HRV_MIN_SIGQ) && !motion_since_last_beat;
              bool delta_ok   = (lastDelta == 0) ||
                                (abs((int)delta - (int)lastDelta) < (int)(RR_TOLERANCE * lastDelta));
              bool rr_ok      = in_range && beat_clean && delta_ok;

              rr_history[rr_index] = rr_ok ? (int)delta : 0;
              rr_index = (rr_index + 1) % MAX_RR_LEN;
              if (rr_count < MAX_RR_LEN) rr_count++;
              calculateHRV();
#if DEBUG_MODE
              if (rr_ok) dbg_clean_beats++; else dbg_gap_beats++;
#endif

              if (in_range) lastDelta = delta;
              motion_since_last_beat = false;
            }
          }
        }
      } else {

        avgBPM               = 0.0f;
        beatsPerMinute       = 0.0f;
        ESpO2                = 0.0;
        spo2_valid           = false;
        avgHRV               = 0.0f;
        hrv_valid            = false;
        signalQuality        = 0.0f;
        finger_detected_time = 0;
        last_warmup_print    = 0;
        rr_count             = 0;
        rr_index             = 0;
        lastDelta            = 0;
        have_last_beat       = false;
        resetBeatDetector();
        motion_since_last_beat = false;
        spo2_windows_seen    = 0;
        bpm_count            = 0;
        dec_red = 0.0; dec_ir = 0.0; dec_n = 0; dec_motion = false;
        dc_slow_ema          = 0.0f;
        ac_envelope          = 0.0f;
        avered               = 0.0;
        aveir                = 0.0;
        aveir_prev_window    = 0.0;
        sumredrms            = 0.0;
        sumirrms             = 0.0;
        spo2_sample_i        = 0;
        spo2_motion_n        = 0;
        for (byte x = 0; x < RATE_SIZE; x++) rates[x] = 0;
        rateSpot = 0;
      }

      particleSensor.nextSample();
    }

  // ---- temperature: refreshed once per second (readTemperature blocks ~30 ms) ----
  if (irValue > FINGER_IR_MIN && (last_temp_t == 0 || millis() - last_temp_t >= TEMP_PERIOD_MS)) {
    current_temp_skin = particleSensor.readTemperature();
    last_temp_t = millis();
  }
  if (irValue <= FINGER_IR_MIN) last_temp_t = 0;

  // ---- one feature row every ROW_PERIOD_MS (~34 Hz) ----
  if (millis() - last_row_t >= ROW_PERIOD_MS) {
    last_row_t = millis();

    bool finger    = irValue > FINGER_IR_MIN;
    bool warmed_up = (finger_detected_time > 0) &&
                     (millis() - finger_detected_time > WARMUP_MS);

    if (!finger) {
      window_rows = 0;
      if (millis() - last_msg_print >= 1000UL) {
        last_msg_print = millis();
        Serial.println("NO FINGER DETECTED");
      }

    } else if (!warmed_up) {
      window_rows = 0;
      if (millis() - last_warmup_print >= 1000UL) {
        last_warmup_print = millis();
        unsigned long elapsed   = millis() - finger_detected_time;
        unsigned long remaining = (WARMUP_MS > elapsed) ?
                                  (WARMUP_MS - elapsed + 999UL) / 1000UL : 0;
        Serial.print("Warming up... ");
        Serial.print(remaining);
        Serial.println("s remaining. Hold still.");
      }

    } else {
      body_temp = current_temp_skin + TEMP_OFFSET;
      temp_sane = (body_temp >= TEMP_MIN_C && body_temp <= TEMP_MAX_C);

      // Only rows made of valid BPM, SpO2 and HRV enter the window (no zero placeholders)
      bool data_ok = (avgBPM > 40.0f) && spo2_valid && hrv_valid;

      if (!data_ok) {
        window_rows = 0;
        if (millis() - last_msg_print >= 2000UL) {
          last_msg_print = millis();
          Serial.println("Waiting for valid BPM / SpO2 / HRV...");
#if DEBUG_MODE
          Serial.printf(" [DBG] BPM %.1f %s | SpO2 %s (R %.3f) | HRV %s | SigQ %.0f%% PI %.2f%% | motion %lu/%lu | RR clean/gap %lu/%lu\n",
                        avgBPM, (avgBPM > 40.0f) ? "ok" : "--",
                        spo2_valid ? "ok" : "--", last_R, hrv_valid ? "ok" : "--",
                        signalQuality * 100.0f, last_pi,
                        dbg_motion_samples, dbg_finger_samples, dbg_clean_beats, dbg_gap_beats);
          printRecentRR();
          dbg_finger_samples = 0; dbg_motion_samples = 0;
          dbg_clean_beats = 0; dbg_gap_beats = 0; dbg_spo2_rejected = 0;
#endif
        }
      } else {
        // sliding window: when full, drop the oldest row
        if (window_rows >= WINDOW_ROWS) {
          memmove(features, features + MODEL_AXES, sizeof(float) * MODEL_AXES * (WINDOW_ROWS - 1));
          window_rows = WINDOW_ROWS - 1;
        }
        float* row = features + MODEL_AXES * window_rows;
        row[0] = avgBPM;
        row[1] = constrain((float)ESpO2 + SPO2_CORRECTION, 70.0f, 100.0f);
        // temperature is not validated: the model gets a neutral value so that an uncalibrated
        // reading (e.g. 35.8 C while the die warms up) cannot drive the class (it gave false Hypoxia)
        row[2] = (TEMP_VALID && temp_sane) ? body_temp : TEMP_NEUTRAL_C;
        row[3] = avgHRV;
        row[4] = pnn50;
        if (MODEL_AXES >= 6) row[5] = signalQuality;
        window_rows++;
      }
    }
  }

  // ---- classify every INFER_STRIDE_MS once the window is full ----
  if (window_rows >= WINDOW_ROWS && millis() - last_infer_t >= INFER_STRIDE_MS) {
    last_infer_t = millis();
    classifyAndReport();
  }
}
