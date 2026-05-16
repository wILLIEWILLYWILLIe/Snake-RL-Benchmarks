// I2C accelerometer/magnetometer app
//
// Read from I2C accelerometer/magnetometer on the Microbit

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "nrf_delay.h"
#include "nrf_twi_mngr.h"

#include "app_timer.h"
#include "microbit_v2.h"
#include "lsm303agr.h"

// Global variables
NRF_TWI_MNGR_DEF(twi_mngr_instance, 1, 0);
APP_TIMER_DEF(my_timer_1);
// APP_TIMER_DEF(my_timer_2);
// APP_TIMER_DEF(my_timer_3);

static void temp_timer_callback(void* context) {
  float temp = lsm303agr_read_temperature();
  printf("Temperature: %.2f C\n", temp);
}
static void accel_timer_callback(void* context) {
  lsm303agr_measurement_t accel = lsm303agr_read_accelerometer();
  printf("Acceleration: X=%.2f g, Y=%.2f g, Z=%.2f g\n", accel.x_axis, accel.y_axis, accel.z_axis);
  lsm303agr_measurement_t angles = acc_to_angle();
  printf("Angles: theta=%.2f, psi=%.2f, phi=%.2f\n", angles.x_axis, angles.y_axis, angles.z_axis);
}
static void mag_timer_callback(void* context) {
  lsm303agr_measurement_t mag = lsm303agr_read_magnetometer();
  printf("Magnetometer: X=%.2f uT, Y=%.2f uT, Z=%.2f uT\n", mag.x_axis, mag.y_axis, mag.z_axis);
}

static void read_ligh_sensor(void* context) {
  while (!bh1750_measurement_ready(false)) {
    // Wait for measurement to be ready
    nrf_delay_ms(10);
    printf("Waiting for light measurement to be ready...\n");
  }
  float level = readLightLevel();
  if (level < 0.0f) {
    printf("Error reading light level: %.2f\n", level);
  } else {
    printf("Light level: %.2f lux\n", level);
  }
}

int main(void) {
  printf("Board started!\n");

  // Initialize I2C peripheral and driver
  nrf_drv_twi_config_t i2c_config = NRF_DRV_TWI_DEFAULT_CONFIG;
  // WARNING!!
  // These are NOT the correct pins for external I2C communication.
  // If you are using QWIIC or other external I2C devices, the are
  // connected to EDGE_P19 (a.k.a. I2C_QWIIC_SCL) and EDGE_P20 (a.k.a. I2C_QWIIC_SDA)
  i2c_config.scl = I2C_INTERNAL_SCL;
  i2c_config.sda = I2C_INTERNAL_SDA;
  i2c_config.frequency = NRF_DRV_TWI_FREQ_100K;
  i2c_config.interrupt_priority = 0;
  nrf_twi_mngr_init(&twi_mngr_instance, &i2c_config);

  // Initialize the LSM303AGR accelerometer/magnetometer sensor
  // lsm303agr_init(&twi_mngr_instance);

  bh1750_init(&twi_mngr_instance);
  set_light_mode(CONTINUOUS_HIGH_RES_MODE);

  app_timer_init();
  app_timer_create(&my_timer_1, APP_TIMER_MODE_REPEATED, read_ligh_sensor);
  app_timer_start(my_timer_1, 32768, NULL);

  // Loop forever
  while (1) {
    // Don't put any code in here. Instead put periodic code in a callback using a timer.
    nrf_delay_ms(1000);
  }
}

