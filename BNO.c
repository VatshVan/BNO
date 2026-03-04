#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

/*
  Hardware Configuration:
  - I2C SDA: GPIO 21 (Default ESP32)
  - I2C SCL: GPIO 22 (Default ESP32)
  - BNO055 I2C Address: 0x28
*/

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);

// Operational parameters
const unsigned long LOOP_PERIOD_MS = 10; // 100 Hz transmission rate
unsigned long last_execution_time = 0;

void setup() {
  Serial.begin(115200);

  // Block execution until serial interface is established
  while (!Serial) {
    delay(10);
  }

  // Initialize the I2C bus and the sensor
  if (!bno.begin()) {
    Serial.println("FATAL ERROR: BNO055 initialization failed. Verify I2C bus continuity.");
    while (1) {
      delay(100); // Halt execution
    }
  }

  // Engage the external crystal oscillator for thermal stability
  bno.setExtCrystalUse(true);

  // Synchronize initial baseline
  last_execution_time = millis();
}

void loop() {
  unsigned long current_time = millis();

  // Non-blocking deterministic temporal gate (100 Hz execution)
  if (current_time - last_execution_time >= LOOP_PERIOD_MS) {
    last_execution_time = current_time;

    // Allocate memory for sensor event structs
    sensors_event_t accelerometer_data;
    sensors_event_t gyroscope_data;

    // Poll the BNO055 registers
    bno.getEvent(&accelerometer_data, Adafruit_BNO055::VECTOR_ACCELEROMETER);
    bno.getEvent(&gyroscope_data, Adafruit_BNO055::VECTOR_GYROSCOPE);

    // Format the 7-element payload vector:
    // timestamp_ms, ax, ay, az, gx, gy, gz
    Serial.print(current_time);
    Serial.print(",");
    
    // Output Linear Acceleration (m/s^2)
    Serial.print(accelerometer_data.acceleration.x, 4);
    Serial.print(",");
    Serial.print(accelerometer_data.acceleration.y, 4);
    Serial.print(",");
    Serial.print(accelerometer_data.acceleration.z, 4);
    Serial.print(",");

    // Output Angular Velocity (rad/s)
    Serial.print(gyroscope_data.gyro.x, 6);
    Serial.print(",");
    Serial.print(gyroscope_data.gyro.y, 6);
    Serial.print(",");
    Serial.print(gyroscope_data.gyro.z, 6);
    
    // Terminate the payload frame
    Serial.println();
  }
}