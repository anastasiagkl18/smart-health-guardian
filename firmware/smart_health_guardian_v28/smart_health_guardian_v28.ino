#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <Anastasia-Health-Project_inferencing.h>

MAX30105 particleSensor;

#define DSP_INPUT_FRAME_SIZE EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE
float features[DSP_INPUT_FRAME_SIZE];


#define DEBUG_MODE 0


#define WARMUP_MS 15000UL


const float TEMP_OFFSET     = 4.3f;   
const bool  TEMP_VALID      = false;  // single-point estimate: NOT validated -> Fever is shown as Uncertain  
const float SPO2_CORRECTION = 4.0f;  


const byte  RATE_SIZE      = 4;
byte        rates[RATE_SIZE];
byte        rateSpot       = 0;
unsigned long        lastBeatClock  = 0;
bool                 have_last_beat = false;
unsigned long        sample_clock_ms = 0;
double               dec_red = 0.0, dec_ir = 0.0;
int                  dec_n = 0;
unsigned long        lastDelta      = 0;
float       beatsPerMinute = 0.0f;
float       avgBPM         = 0.0f;
const float BPM_ALPHA      = 0.10f;


double avered        = 0.0;
double aveir         = 0.0;
double sumirrms      = 0.0;
double sumredrms     = 0.0;
int    spo2_sample_i = 0;
double aveir_prev_window = 0.0;  
const  int    SPO2_SAMPLE_NUM = 50;   // 50 samples @25 Hz = 2 s window
const  int    SPO2_DECIM      = 4;    // 100 Hz -> 25 Hz for SpO2
const  unsigned long SAMPLE_PERIOD_MS = 10; // 100 Hz sample clock
const  float  DRIFT_THRESHOLD = 0.01f;
int    spo2_windows_seen = 0;   // first windows after finger-on are discarded
byte   bpm_count = 0;           // valid beat intervals since finger-on
double ESpO2         = 0.0;
const  double FSpO2  = 0.85;
const  double frate  = 0.95;   // tau ~ 20 samples @25 Hz = 0.8 s
bool   spo2_valid    = false;


const int MAX_RR_LEN    = 20;
const int MIN_HRV_PAIRS = 5;

int   rr_history[MAX_RR_LEN];
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
const float AC_ENV_ALPHA = 0.10f; 
const float MOTION_THRESHOLD = 0.02f;
const long  FINGER_IR_MIN        = 7000;    // IR counts above which a finger is present
const float RR_TOLERANCE         = 0.18f;   // max relative change between successive RR
const float MIN_CONFIDENCE       = 0.60f;   // below this the result is shown as Uncertain
const float ARRHYTHMIA_MIN_CONF  = 0.80f;   // Arrhythmia needs high model confidence...
const float ARRHYTHMIA_MIN_SIGQ  = 0.30f;   // ...and acceptable signal quality
const float TEMP_MIN_C           = 35.0f;   // plausible body-temperature window
const float TEMP_MAX_C           = 42.0f;
float body_temp  = 0.0f;
float current_temp_skin = 0.0f;               // last die-temperature reading (refreshed every TEMP_PERIOD_MS)
int   window_rows = 0;                        // rows currently in the sliding window
const unsigned long ROW_PERIOD_MS   = 29;     // ~34 Hz, matches the Edge Impulse impulse (34 Hz)
const unsigned long INFER_STRIDE_MS = 1000;   // classify every 1 s -> same stride as the impulse
const unsigned long TEMP_PERIOD_MS  = 1000;   // temperature changes slowly: read it once per second
bool  temp_sane  = false;
long  irValue    = 0;
long  redValue   = 0;


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


  bool is_motion = (abs_ac / dc_slow_ema) > MOTION_THRESHOLD;

  if (!is_motion) {
    ac_envelope = AC_ENV_ALPHA * abs_ac + (1.0f - AC_ENV_ALPHA) * ac_envelope;
  }
 

  if (dc_settle_count < DC_SETTLE_SAMPLES + (int)(1.5f / AC_ENV_ALPHA)) {
    signalQuality = 0.0f;
    return;
  }

  
  float pi = (ac_envelope / dc_slow_ema) * 100.0f;

 
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

    if (RR_prev == 0 || RR_now == 0) continue;

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

  Serial.println("\n-------------------------------------------");
  Serial.println(" SMART HEALTH GUARDIAN - V28.0            ");
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

  Serial.println();

  if (avgBPM > 40) {
   
    // Display rules (the model output itself is never modified):
    //  - Arrhythmia needs SigQ >= ARRHYTHMIA_MIN_SIGQ and confidence >= ARRHYTHMIA_MIN_CONF
    //  - Fever is not reported while the temperature estimate is not validated
    //  - any result below MIN_CONFIDENCE is shown as Uncertain
    const char* display_label = ai_label;
    const char* reason        = nullptr;
    if (strcmp(ai_label, "Arrhythmia") == 0 &&
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
  const int WINDOW_ROWS = DSP_INPUT_FRAME_SIZE / 6;

    particleSensor.check();

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

      
      bool motion_ok = (dc_slow_ema > 1000.0f) &&
                       (fabsf((float)fir_raw - dc_slow_ema) / dc_slow_ema < MOTION_THRESHOLD);

      
      // SpO2 stream: average groups of SPO2_DECIM samples -> 25 Hz
      dec_red += fred; dec_ir += fir; dec_n++;
      if (dec_n >= SPO2_DECIM) {
        double fr = dec_red / dec_n;
        double fi = dec_ir  / dec_n;
        dec_red = 0.0; dec_ir = 0.0; dec_n = 0;

        if (avered == 0.0) { avered = fr; aveir = fi; }
        avered = avered * frate + fr * (1.0 - frate);
        aveir  = aveir  * frate + fi * (1.0 - frate);

        if (!motion_ok) {
          sumredrms = 0.0; sumirrms = 0.0; spo2_sample_i = 0;
        } else {
          sumredrms += (fr - avered) * (fr - avered);
          sumirrms  += (fi - aveir)  * (fi - aveir);
          spo2_sample_i++;
        }

        if (spo2_sample_i > 0 && (spo2_sample_i % SPO2_SAMPLE_NUM) == 0) {
          bool drift_ok = true;
          if (aveir_prev_window > 0.0) {
            drift_ok = (fabs(aveir - aveir_prev_window) / aveir_prev_window < DRIFT_THRESHOLD);
          }
          aveir_prev_window = aveir;

          spo2_windows_seen++;
          if (motion_ok && drift_ok && spo2_windows_seen > 2 && avered > 0 && aveir > 0 && sumirrms > 0) {
            double R = (sqrt(sumredrms) / avered) / (sqrt(sumirrms) / aveir);
            R = constrain(R, 0.4, 1.8);
            double SpO2 = 104.0 - 17.0 * R;
            SpO2 = constrain(SpO2, 70.0, 100.0);

            if (ESpO2 == 0.0) ESpO2 = SpO2;
            else              ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2;

            spo2_valid = (ESpO2 >= 70.0);
          }
          sumredrms = 0.0; sumirrms = 0.0; spo2_sample_i = 0;
        }
      }

      { static int sig_skip = 0;
        if (++sig_skip >= 5) { sig_skip = 0; updateSignalQuality(irValue); } }

      
      if (irValue > FINGER_IR_MIN) {
        if (finger_detected_time == 0) finger_detected_time = millis();

        if (checkForBeat(irValue)) {
          if (!have_last_beat) {
            have_last_beat = true;
            lastBeatClock  = sample_clock_ms;
          } else {
            unsigned long delta = sample_clock_ms - lastBeatClock;

            // adaptive refractory: rejects the dicrotic notch at rest, allows tachycardia
            unsigned long min_gap = (lastDelta > 0) ? 300UL : 400UL;
            if (lastDelta > 0) {
              unsigned long adaptive = (unsigned long)(0.6f * lastDelta);
              if (adaptive > min_gap) min_gap = adaptive;
            }

            if (delta >= min_gap) {
              lastBeatClock  = sample_clock_ms;
              beatsPerMinute = 60000.0f / (float)delta;

              if (beatsPerMinute > 40 && beatsPerMinute < 200) {
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

              if (delta > 300 && delta < 1500) {
                bool delta_ok = (lastDelta == 0) ||
                                (abs((int)delta - (int)lastDelta) < (int)(RR_TOLERANCE * lastDelta));
                if (delta_ok) {
                  rr_history[rr_index] = (int)delta;
                  rr_index = (rr_index + 1) % MAX_RR_LEN;
                  if (rr_count < MAX_RR_LEN) rr_count++;
                  calculateHRV();
                }
                lastDelta = delta;
              }
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
        spo2_windows_seen    = 0;
        bpm_count            = 0;
        dec_red = 0.0; dec_ir = 0.0; dec_n = 0;
        dc_slow_ema          = 0.0f;
        ac_envelope          = 0.0f;
        avered               = 0.0;
        aveir                = 0.0;
        aveir_prev_window    = 0.0;
        sumredrms            = 0.0;
        sumirrms             = 0.0;
        spo2_sample_i        = 0;
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
        }
      } else {
        // sliding window: when full, drop the oldest row
        if (window_rows >= WINDOW_ROWS) {
          memmove(features, features + 6, sizeof(float) * 6 * (WINDOW_ROWS - 1));
          window_rows = WINDOW_ROWS - 1;
        }
        float* row = features + 6 * window_rows;
        row[0] = avgBPM;
        row[1] = constrain((float)ESpO2 + SPO2_CORRECTION, 70.0f, 100.0f);
        row[2] = temp_sane ? body_temp : 37.0f;
        row[3] = avgHRV;
        row[4] = pnn50;
        row[5] = signalQuality;
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
