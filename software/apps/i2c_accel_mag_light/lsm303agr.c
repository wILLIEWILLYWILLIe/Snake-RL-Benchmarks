// LSM303AGR driver for Microbit_v2
//
// Initializes sensor and communicates over I2C
// Capable of reading temperature, acceleration, and magnetic field strength

#include <stdbool.h>
#include <stdint.h>
#include <math.h> 

#include "app_timer.h"
#include "lsm303agr.h"
#include "nrf_delay.h"

// Pointer to an initialized I2C instance to use for transactions
static const nrf_twi_mngr_t* i2c_manager = NULL;

static const uint8_t BH1750_ADDR = 0x23;
static Light_Mode BH1750_MODE = UNCONFIGURED;
static uint8_t BH1750_MTreg = BH1750_DEFAULT_MTREG;
static uint32_t last_read_ticks = 0;

// Helper function to perform a 1-byte I2C read of a given register
//
// i2c_addr - address of the device to read from
// reg_addr - address of the register within the device to read
//
// returns 8-bit read value
static uint8_t i2c_reg_read(uint8_t i2c_addr, uint8_t reg_addr) {
  uint8_t rx_buf = 0;
  nrf_twi_mngr_transfer_t const read_transfer[] = {
    NRF_TWI_MNGR_WRITE(i2c_addr, &reg_addr, 1, NRF_TWI_MNGR_NO_STOP),
    NRF_TWI_MNGR_READ(i2c_addr, &rx_buf, 1, 0),
  };
  ret_code_t result = nrf_twi_mngr_perform(i2c_manager, NULL, read_transfer, 2, NULL);
  if (result != NRF_SUCCESS) {
    // Likely error codes:
    //  NRF_ERROR_INTERNAL            (0x0003) - something is wrong with the driver itself
    //  NRF_ERROR_INVALID_ADDR        (0x0010) - buffer passed was in Flash instead of RAM
    //  NRF_ERROR_BUSY                (0x0011) - driver was busy with another transfer still
    //  NRF_ERROR_DRV_TWI_ERR_OVERRUN (0x8200) - data was overwritten during the transaction
    //  NRF_ERROR_DRV_TWI_ERR_ANACK   (0x8201) - i2c device did not acknowledge its address
    //  NRF_ERROR_DRV_TWI_ERR_DNACK   (0x8202) - i2c device did not acknowledge a data byte
    printf("I2C transaction failed! Error: %lX\n", result);
  }

  return rx_buf;
}

// Helper function to perform a 1-byte I2C write of a given register
//
// i2c_addr - address of the device to write to
// reg_addr - address of the register within the device to write
static void i2c_reg_write(uint8_t i2c_addr, uint8_t reg_addr, uint8_t data) {
  uint8_t tx_buf[2] = {reg_addr, data};
  nrf_twi_mngr_transfer_t const write_transfer[] = {
    NRF_TWI_MNGR_WRITE(i2c_addr, tx_buf, 2, 0),
  };
  ret_code_t result = nrf_twi_mngr_perform(i2c_manager, NULL, write_transfer, 1, NULL);
}

// Initialize and configure the LSM303AGR accelerometer/magnetometer
//
// i2c - pointer to already initialized and enabled twim instance
void lsm303agr_init(const nrf_twi_mngr_t* i2c) {
  i2c_manager = i2c;

  // ---Initialize Accelerometer---

  // Reboot acclerometer
  i2c_reg_write(LSM303AGR_ACC_ADDRESS, CTRL_REG5_A, 0x80);
  nrf_delay_ms(100); // needs delay to wait for reboot

  // Enable Block Data Update
  // Only updates sensor data when both halves of the data has been read
  i2c_reg_write(LSM303AGR_ACC_ADDRESS, CTRL_REG4_A, 0x80);

  // Configure accelerometer at 100Hz, normal mode (10-bit)
  // Enable x, y and z axes
  i2c_reg_write(LSM303AGR_ACC_ADDRESS, CTRL_REG1_A, 0x57);

  // Read WHO AM I register
  // Always returns the same value if working
  uint8_t accel_whoami = i2c_reg_read(LSM303AGR_ACC_ADDRESS, WHO_AM_I_A);
  printf("Accelerometer WHO AM I: 0x%X \n", accel_whoami);

  // ---Initialize Magnetometer---

  // Reboot magnetometer
  i2c_reg_write(LSM303AGR_MAG_ADDRESS, CFG_REG_A_M, 0x40);
  nrf_delay_ms(100); // needs delay to wait for reboot

  // Enable Block Data Update
  // Only updates sensor data when both halves of the data has been read
  i2c_reg_write(LSM303AGR_MAG_ADDRESS, CFG_REG_C_M, 0x10);

  // Configure magnetometer at 100Hz, continuous mode
  i2c_reg_write(LSM303AGR_MAG_ADDRESS, CFG_REG_A_M, 0x0C);

  // Read WHO AM I register
  uint8_t mag_whoami = i2c_reg_read(LSM303AGR_MAG_ADDRESS, WHO_AM_I_M);
  printf("Magnetometer WHO AM I: 0x%X \n", mag_whoami);

  // ---Initialize Temperature---

  // Enable temperature sensor
  i2c_reg_write(LSM303AGR_ACC_ADDRESS, TEMP_CFG_REG_A, 0xC0);
  nrf_delay_ms(100); // needs delay to be ready
}

// Read the internal temperature sensor
//
// Return measurement as floating point value in degrees C
float lsm303agr_read_temperature(void) {
  uint8_t temp_l = i2c_reg_read(LSM303AGR_ACC_ADDRESS, OUT_TEMP_L_A);
  uint8_t temp_h = i2c_reg_read(LSM303AGR_ACC_ADDRESS, OUT_TEMP_H_A);
  int16_t raw = (int16_t)((temp_h << 8) | temp_l);
  return (float)raw * (1.0f / 256.0f) + 25.0f;
}

lsm303agr_measurement_t lsm303agr_read_accelerometer(void) {
  int16_t x_raw = (int16_t)((i2c_reg_read(LSM303AGR_ACC_ADDRESS, OUT_X_H_A) << 8) |
                              i2c_reg_read(LSM303AGR_ACC_ADDRESS, OUT_X_L_A));
  int16_t y_raw = (int16_t)((i2c_reg_read(LSM303AGR_ACC_ADDRESS, OUT_Y_H_A) << 8) |
                              i2c_reg_read(LSM303AGR_ACC_ADDRESS, OUT_Y_L_A));
  int16_t z_raw = (int16_t)((i2c_reg_read(LSM303AGR_ACC_ADDRESS, OUT_Z_H_A) << 8) |
                              i2c_reg_read(LSM303AGR_ACC_ADDRESS, OUT_Z_L_A));

  lsm303agr_measurement_t measurement = {
    .x_axis = (float)(x_raw >> 6) * 0.0039f,
    .y_axis = (float)(y_raw >> 6) * 0.0039f,
    .z_axis = (float)(z_raw >> 6) * 0.0039f,
  };
  return measurement;
}

lsm303agr_measurement_t lsm303agr_read_magnetometer(void) {
  //TODO: implement me
  int16_t x_raw = (int16_t)((i2c_reg_read(LSM303AGR_MAG_ADDRESS, OUTX_H_REG_M) << 8) |
                              i2c_reg_read(LSM303AGR_MAG_ADDRESS, OUTX_L_REG_M));
  int16_t y_raw = (int16_t)((i2c_reg_read(LSM303AGR_MAG_ADDRESS, OUTY_H_REG_M) << 8) |
                              i2c_reg_read(LSM303AGR_MAG_ADDRESS, OUTY_L_REG_M));
  int16_t z_raw = (int16_t)((i2c_reg_read(LSM303AGR_MAG_ADDRESS, OUTZ_H_REG_M) << 8) |
                              i2c_reg_read(LSM303AGR_MAG_ADDRESS, OUTZ_L_REG_M));

  lsm303agr_measurement_t measurement = {
    .x_axis = (float)x_raw * 0.1f,
    .y_axis = (float)y_raw * 0.1f,
    .z_axis = (float)z_raw * 0.1f,
  };

  return measurement;
}

lsm303agr_measurement_t acc_to_angle(void) {
  lsm303agr_measurement_t accel = lsm303agr_read_accelerometer();
  float theta = atan((accel.x_axis)/sqrt(pow(accel.y_axis, 2) + pow(accel.z_axis, 2)));
  float psi = atan((accel.y_axis)/sqrt(pow(accel.x_axis, 2) + pow(accel.z_axis, 2)));
  float phi = atan(sqrt(pow(accel.x_axis, 2) + pow(accel.y_axis, 2))/accel.z_axis);

  lsm303agr_measurement_t measurement = {
    .x_axis = (float)theta * (180.0f / M_PI),
    .y_axis = (float)psi * (180.0f / M_PI),
    .z_axis = (float)phi * (180.0f / M_PI),
  };

  return measurement;
}


// Initialize BH1750 light sensor in continuous high-resolution mode

void set_light_mode(Light_Mode mode) {
  uint8_t cmd = (uint8_t)mode;
  nrf_twi_mngr_transfer_t const set_mode[] = {
    NRF_TWI_MNGR_WRITE(BH1750_ADDR, &cmd, 1, 0),
  };
  nrf_twi_mngr_perform(i2c_manager, NULL, set_mode, 1, NULL);
  BH1750_MODE = mode;

  // Wait for measurement time to be ready
  if (mode == CONTINUOUS_HIGH_RES_MODE || mode == ONE_TIME_HIGH_RES_MODE) {
    nrf_delay_ms(180);
  } else if (mode == CONTINUOUS_HIGH_RES_MODE_2 || mode == ONE_TIME_HIGH_RES_MODE_2) {
    nrf_delay_ms(180);
  } else if (mode == CONTINUOUS_LOW_RES_MODE || mode == ONE_TIME_LOW_RES_MODE) {
    nrf_delay_ms(24);
  }
}

bool bh1750_set_mtreg(uint8_t MTreg) {
  if (MTreg < BH1750_MTREG_MIN || MTreg > BH1750_MTREG_MAX) {
    printf("BH1750: MTreg out of range\n");
    return false;
  }

  // Split MTreg across two command bytes per BH1750 protocol:
  //   High byte: 01000_MT[7,6,5]
  //   Low byte:  011_MT[4,3,2,1,0]
  uint8_t high = (0b01000 << 3) | (MTreg >> 5);
  uint8_t low  = (0b011   << 5) | (MTreg & 0b11111);
  uint8_t mode = (uint8_t)BH1750_MODE;

  nrf_twi_mngr_transfer_t const t_high[] = { NRF_TWI_MNGR_WRITE(BH1750_ADDR, &high, 1, 0) };
  nrf_twi_mngr_transfer_t const t_low[]  = { NRF_TWI_MNGR_WRITE(BH1750_ADDR, &low,  1, 0) };
  nrf_twi_mngr_transfer_t const t_mode[] = { NRF_TWI_MNGR_WRITE(BH1750_ADDR, &mode, 1, 0) };
  ret_code_t result = nrf_twi_mngr_perform(i2c_manager, NULL, t_high, 1, NULL);
  result |= nrf_twi_mngr_perform(i2c_manager, NULL, t_low,  1, NULL);
  result |= nrf_twi_mngr_perform(i2c_manager, NULL, t_mode, 1, NULL);
  nrf_delay_ms(10);

  if (result != NRF_SUCCESS) {
    printf("BH1750: set_mtreg failed! Error: %lX\n", result);
    return false;
  }

  BH1750_MTreg = MTreg;
  return true;
}




void bh1750_init(const nrf_twi_mngr_t* i2c) {
  i2c_manager = i2c;
  uint8_t cmd = BH1750_POWER_ON;
  nrf_twi_mngr_transfer_t const power_on[] = {
    NRF_TWI_MNGR_WRITE(BH1750_ADDR, &cmd, 1, 0),
  };
  nrf_twi_mngr_perform(i2c_manager, NULL, power_on, 1, NULL);
}


// Returns true if enough time has elapsed since the last read for a new
// measurement to be ready. max_wait=true uses the maximum measurement time;
// false uses the typical (shorter) time. See datasheet pages 2, 5, 7.
bool bh1750_measurement_ready(bool max_wait) {
  uint32_t delaytime_ms = 0;
  switch (BH1750_MODE) {
  case CONTINUOUS_HIGH_RES_MODE:
  case CONTINUOUS_HIGH_RES_MODE_2:
  case ONE_TIME_HIGH_RES_MODE:
  case ONE_TIME_HIGH_RES_MODE_2:
    delaytime_ms = max_wait ? (180 * BH1750_MTreg / BH1750_DEFAULT_MTREG)
                            : (120 * BH1750_MTreg / BH1750_DEFAULT_MTREG);
    break;
  case CONTINUOUS_LOW_RES_MODE:
  case ONE_TIME_LOW_RES_MODE:
    delaytime_ms = max_wait ? (24 * BH1750_MTreg / BH1750_DEFAULT_MTREG)
                            : (16 * BH1750_MTreg / BH1750_DEFAULT_MTREG);
    break;
  default:
    break;
  }
  uint32_t elapsed = app_timer_cnt_diff_compute(app_timer_cnt_get(), last_read_ticks);
  return elapsed >= APP_TIMER_TICKS(delaytime_ms);
}


// Read light level from BH1750 in lux
float readLightLevel(void) {
  if (BH1750_MODE == UNCONFIGURED) {
    printf("BH1750: not configured!\n");
    return -2.0f;
  }

  uint8_t rx_buf[2] = {0};
  nrf_twi_mngr_transfer_t const read_transfer[] = {
    NRF_TWI_MNGR_READ(BH1750_ADDR, rx_buf, 2, 0),
  };
  ret_code_t result = nrf_twi_mngr_perform(i2c_manager, NULL, read_transfer, 1, NULL);
  if (result != NRF_SUCCESS) {
    return -1.0f;
  }
  last_read_ticks = app_timer_cnt_get();

  uint16_t raw = ((uint16_t)rx_buf[0] << 8) | rx_buf[1];
  float level = (float)raw;

  if (BH1750_MTreg != BH1750_DEFAULT_MTREG) {
    level *= (float)BH1750_DEFAULT_MTREG / (float)BH1750_MTreg;
  }
  if (BH1750_MODE == ONE_TIME_HIGH_RES_MODE_2 || BH1750_MODE == CONTINUOUS_HIGH_RES_MODE_2) {
    level /= 2.0f;
  }
  level /= 1.2f;

  return level;
}
